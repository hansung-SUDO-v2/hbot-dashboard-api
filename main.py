from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from apps.clustering import initialize, run_clustering
from apps.config import APP_HOST, APP_PORT
from apps.routes import router

app = FastAPI()
app.include_router(router)

initialize()

scheduler = BackgroundScheduler()
scheduler.add_job(run_clustering, "cron", hour=0, minute=0)
scheduler.start()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=APP_HOST, port=APP_PORT, reload=True)
