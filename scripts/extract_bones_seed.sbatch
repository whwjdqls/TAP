#!/bin/bash
#SBATCH --job-name=bones_extract
#SBATCH --partition=pilab
#SBATCH --qos=qos-pi-top
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/nfsdata/home/jungbin.cho/TAP/runs/bones_extract_%j.out
#SBATCH --error=/nfsdata/home/jungbin.cho/TAP/runs/bones_extract_%j.err

set -eo pipefail
echo "Job ${SLURM_JOB_ID:-local} started on $(hostname) at $(date)"

cd /nfsdata/home/jungbin.cho/TAP/data/bones_seed
test -f g1.tar.gz || { echo "missing g1.tar.gz" >&2; exit 1; }
echo "tarball size: $(du -h g1.tar.gz | cut -f1)"

# Use pigz if available for parallel decompression; fall back to gzip.
if command -v pigz >/dev/null 2>&1; then
  echo "using pigz"
  pigz -dc g1.tar.gz | tar -xf -
else
  echo "using gzip"
  tar -xzf g1.tar.gz
fi

echo "csv count: $(find g1 -type f -name '*.csv' | wc -l)"
echo "extracted size: $(du -sh g1 | cut -f1)"
echo "Job finished at $(date)"
