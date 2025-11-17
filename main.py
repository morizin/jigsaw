from src.jigsaw.pipelines.base import BasePipeline

if __name__ == "__main__":
    try:
        pipeline = BasePipeline()
        pipeline.kickoff()
    except Exception as e:
        raise e
