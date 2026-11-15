"""Optional, persistent request budget for a single notebook run."""
import contextlib
import contextvars
import json
from threading import RLock
from decimal import Decimal

from .persistence import Store, atomic_json
from .usage import RATES, usage_row

ACTIVE_BUDGET = contextvars.ContextVar('api_budget', default=None)


class BudgetExceeded(RuntimeError):
    """Stop immediately, outside the pipeline's retryable provider errors."""


def reservation_usd(settings, body):
    rates = RATES.get((settings['provider'], settings['model']))
    if rates is None:
        raise BudgetExceeded('No verified pricing for this model; refusing a budgeted API call')
    reservation = Decimal(0)
    if body is not None:
        if settings['provider'] != 'openai' or body.get('service_tier') != 'default':
            raise BudgetExceeded('Budgeted calls require OpenAI Standard processing')
        # Conservative text-only token allowance: UTF-8 bytes plus framing.
        # No cache discount is assumed; allow the higher cache-write rate.
        input_bound = len(json.dumps(body, ensure_ascii=False).encode('utf-8')) + 4096
        i, _, o, w = map(Decimal, rates)
        if input_bound > 272000:
            i, w, o = i * 2, w * 2, o * Decimal('1.5')
        reservation = (input_bound * max(i, w) + body['max_output_tokens'] * o) / 1000000
    return reservation


class APIBudget:
    def __init__(self, directory, max_calls=120, max_cost_usd='0.50'):
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('max_calls must be a positive integer')
        cost = Decimal(str(max_cost_usd))
        if not cost.is_finite() or cost <= 0:
            raise ValueError('max_cost_usd must be finite and positive')
        self.store = Store(directory)
        self._mutex = RLock()
        self._active = False
        self.limits = {'max_calls': max_calls, 'max_cost_usd': str(cost)}

    @contextlib.contextmanager
    def activate(self, *, allow_increase=False):
        # A separate folder/lock prevents two notebooks sharing the same budget.
        with self.store.lock():
            path = self.store.path('state.json')
            if not path.exists() and any(self.store.directory.parent.rglob('api_usage.jsonl')):
                raise ValueError('Existing API usage has no budget state; use a fresh run directory')
            self.state = json.loads(path.read_text()) if path.exists() else {
                'limits': self.limits, 'calls': 0, 'accounted_usd': '0', 'unknown_usage_calls': 0}
            if ACTIVE_BUDGET.get() is not None:
                raise ValueError('Nested API budgets are not supported')
            previous = self.state['limits']
            if (previous['max_calls'] != self.limits['max_calls']
                    or Decimal(previous['max_cost_usd']) != Decimal(self.limits['max_cost_usd'])):
                if not allow_increase:
                    raise ValueError('Budget limits changed; explicitly allow an increase to resume this run')
                if (self.limits['max_calls'] < previous['max_calls']
                        or Decimal(self.limits['max_cost_usd']) < Decimal(previous['max_cost_usd'])):
                    raise ValueError('Resuming with a budget increase cannot lower an existing limit')
                self.state['limits'] = self.limits
                atomic_json(path, self.state)
            if not path.exists():
                atomic_json(path, self.state)
            token = ACTIVE_BUDGET.set(self)
            with self._mutex:
                self._active = True
            try:
                yield self
            finally:
                with self._mutex:
                    self._active = False
                ACTIVE_BUDGET.reset(token)

    def reserve(self, settings, body):
        reservation = reservation_usd(settings, body)
        with self._mutex:
            self._require_active()
            accounted = Decimal(self.state['accounted_usd'])
            if self.state['calls'] >= self.limits['max_calls']:
                raise BudgetExceeded(f"API request limit reached ({self.state['calls']}/{self.limits['max_calls']}). "
                                     'Raise the request limit to resume the same run.')
            cost_limit = Decimal(self.limits['max_cost_usd'])
            if accounted + reservation > cost_limit:
                raise BudgetExceeded(f'API cost limit reached (${accounted:.4f} accounted + '
                                     f'${reservation:.4f} reserved for the next request, limit ${cost_limit:.2f}). '
                                     'Raise the cost limit to resume the same run.')
            # Persist BEFORE sending: a crash or unknown response retains its allowance.
            self._save_state({**self.state, 'calls': self.state['calls'] + 1,
                              'unknown_usage_calls': self.state['unknown_usage_calls'] + 1,
                              'accounted_usd': str(accounted + reservation)})
            return reservation

    def settle(self, reservation, event):
        row = usage_row(event)
        if 'estimated_cost_usd' not in row:
            return  # Unknown failures retain the conservative reservation.
        with self._mutex:
            self._require_active()
            self._save_state({**self.state,
                              'accounted_usd': str(Decimal(self.state['accounted_usd']) - reservation
                                                   + Decimal(row['estimated_cost_usd'])),
                              'unknown_usage_calls': self.state['unknown_usage_calls'] - 1})

    def _require_active(self):
        if not self._active:
            raise BudgetExceeded('API budget is inactive; no request can be reserved or settled')

    def _save_state(self, state):
        atomic_json(self.store.path('state.json'), state)
        self.state = state
