# Docker Usage

Follow these steps to build the PlantCaduceus runtime container, run commands, and validate the setup with the zero-shot mutation effect scoring workflow.

## Build Image

Clone the repository and build the Docker image from the project root:

```bash
git clone --single-branch https://github.com/plantcad/PlantCaduceus.git
cd PlantCaduceus

docker build --progress=plain \
  -f docker/python3.11-torch2.5.1-cuda12.4/Dockerfile -t plantcad-v1.0.0 . 
docker build --progress=plain \
  -f docker/python3.11-torch2.7.1-cuda12.8/Dockerfile -t plantcad-v1.1.0 . 
```

## Test an Image

The following commands reproduce the "Basic usage with VCF" example for zero-shot mutation effect scoring:

```bash
# Run the container
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace plantcad-v1.0.0 

# In the container, download an example reference genome
wget https://download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/Zm-B73-REFERENCE-NAM-5.0.fa.gz
gunzip Zm-B73-REFERENCE-NAM-5.0.fa.gz

# Run the zero-shot scoring workflow
python src/zero_shot_score.py \
    -input-vcf examples/example_maize_snp.vcf \
    -input-fasta Zm-B73-REFERENCE-NAM-5.0.fa \
    -output scored_variants.vcf \
    -model 'kuleshov-group/PlantCaduceus_l32' \
    -device 'cuda:0'
```

The generated `scored_variants.vcf` file contains PlantCaduceus log-likelihood scores for each variant.

## Publish an Image

Retag and push an image to GitHub Container Registry:

```bash
IMAGE=ghcr.io/plantcad/plantcad
VERSION=v1.0.0
docker tag plantcad-$VERSION $IMAGE:$VERSION
# Requires a GitHub personal access token with "write:packages" stored in GITHUB_TOKEN
# GITHUB_TOKEN=<your-token>
# GITHUB_USERNAME=<your-username>
echo $GITHUB_TOKEN | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
docker push $IMAGE:$VERSION
```

Test the published image:

```bash
docker run --rm -it --gpus all -v $(pwd):/workspace -w /workspace \
  ghcr.io/plantcad/plantcad:v1.0.0 python -c "import torch; print(torch.cuda.is_available())"
```
