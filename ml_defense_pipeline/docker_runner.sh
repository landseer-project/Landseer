docker run -it --rm \
  -v "$(pwd)":/app \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -w /app \
  -e HOST_SCRIPTS=/share/landseer/workspace-ayushi/Landseer/ml_defense_pipeline/scripts \
  -e HOST_DATA=/share/landseer/workspace-ayushi/Landseer/ml_defense_pipeline/data \
  python:3.10-slim \
  bash -c "pip install -r requirements.txt && bash"
