from src.jigsaw.pipelines.base import BasePipelines

if __name__ == "__main__":
    try:
        pipeline = BasePipelines()
        pipeline.kickoff()
    except Exception as e:
        raise e
