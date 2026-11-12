# Run the 1,000-pair experiment on EC2

The EC2 launcher uses the Kaggle training profile with Unsloth, 10 rounds,
60 general and 40 adaptive pairs per round, 100 development cases, and the
final holdout. Fresh EC2 runs default to inference batch size 4. Training
still uses one GPU and the same effective batch size as Kaggle.

## Launch the instance

1. In the EC2 console, use the region where your GPU quota is available.
   The configured launch form uses Ohio (`us-east-2`). Check that
   **Running On-Demand G and VT instances** has at least 4 vCPUs available
   in that region.
2. Launch an On-Demand **g6e.xlarge**. Select the Amazon-owned **Deep Learning
   OSS Nvidia Driver AMI GPU PyTorch 2.14 (Ubuntu 26.04)**, x86-64. Use a
   100 GiB gp3 root volume. Create or select an SSH key pair and allow inbound
   SSH from **My IP**. Enable a public IP if you are connecting directly
   from your computer.
3. In **EC2 → Instances → Connect → SSH client**, copy the SSH command
   and note the public DNS name. Download the matching
   `outputs/colab_distillation_bundle.zip` from the workspace to your computer.

From your computer, change into the project directory, substitute your key
path and instance DNS name, and run:

```bash
chmod 400 ~/Downloads/ec2-key.pem
scp -i ~/Downloads/ec2-key.pem outputs/colab_distillation_bundle.zip ubuntu@EC2_PUBLIC_DNS:~/
ssh -i ~/Downloads/ec2-key.pem ubuntu@EC2_PUBLIC_DNS
```

## Install and run

On the instance, check the driver and install the project into the AMI's
PyTorch environment. The pip constraint keeps the CUDA-enabled PyTorch
version supplied by the AMI.

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y tmux unzip
unzip -o ~/colab_distillation_bundle.zip -d ~
cd ~/Health-Insurance-SLM-Colab
source /opt/pytorch/bin/activate
python -c 'from importlib.metadata import version; print("torch==" + version("torch"))' > /tmp/distillation-torch-constraint.txt
python -m pip install -c /tmp/distillation-torch-constraint.txt -r requirements-colab.txt
```

Start a tmux session. Inside it, enter your OpenAI key and your chosen total
teacher API cap. If GitHub Releases are private, also set `GITHUB_TOKEN` using
the same hidden prompt pattern and a token with Contents read access.

```bash
tmux new -s distillation
```

```bash
source /opt/pytorch/bin/activate
cd ~/Health-Insurance-SLM-Colab
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
read -rp 'Total teacher API budget (USD): ' API_MAX_COST_USD
RUN_DIR="$HOME/distillation-runs/aws-1000"
mkdir -p "$RUN_DIR"
set -o pipefail
python -u -m distillation_loop.ec2 \
  --run-dir "$RUN_DIR" \
  --api-max-cost-usd "$API_MAX_COST_USD" \
  2>&1 | tee -a "$RUN_DIR/run.log"
```

Detach with **Ctrl+B**, then **D**. After reconnecting over SSH, reattach with
`tmux attach -t distillation`. The SSH connection can close while the job runs.
The launcher loads the newest compatible GitHub release adapter on a fresh run
and uses the saved local checkpoint on resume. If the teacher budget is
exhausted, rerun with the same run directory and a higher cumulative cap.
If inference at batch size 4 is too large, start a fresh run with
`--inference-batch-size 1`. Keep batch size fixed when resuming an existing run.

To migrate an in-progress Kaggle run, copy the whole run directory to the same
absolute path on EC2 and use that path as `--run-dir`. Use
`--inference-batch-size 1` if that was the saved Kaggle batch size and keep the
same bundle version and teacher budget limits. A GitHub release warm-start
carries adapter weights and starts a new round journal.

The terminal shows the GPU, generation and verification progress, training loss,
round scores and timings, checkpoint paths, and API usage. You can also follow
the same output from a second SSH window with:

```bash
tail -f ~/distillation-runs/aws-1000/run.log
```

The run log, adapters, and evaluation results are in the run directory.
Copy them off the instance when done. Stop the instance to stop compute billing
while preserving its EBS volume. Terminate it after backing up the results if
you no longer need it. EBS storage can continue to incur charges after stopping.
