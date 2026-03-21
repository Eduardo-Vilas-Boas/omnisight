# Step 1: Start the infrastructure (MLflow + Postgres + serving API)
# The -d flag runs everything in the background
docker compose up -d

# Step 2: Open the MLflow UI to see past experiments
xdg-open http://localhost:5000

# Step 3: Check that the serving API is healthy
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Step 4: Run a training job (logs to MLflow)
# -d runs the container in detached mode so it doesn't block your terminal
# --rm cleans up the container automatically when the job finishes
# Check the MLflow UI to see the new run appear and track its progress
docker compose --profile train run -d --rm training

# Step 5: Watch training progress in the MLflow UI at localhost:5000
# You'll see metrics updating in real time if your training code logs them

# Step 6: After training completes, promote the model in MLflow UI
# Open the registered model, go to the version, and set the alias to "champion"
# (MLflow v3 uses aliases instead of stages)

# Step 7: Restart the serving container to pick up the new model version
docker compose restart serving

# Step 8: Test the new model with a video file
curl -X POST http://localhost:8000/analyze \
  -F "file=@/path/to/test_video.mp4"
