FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .


RUN python3.12 -m pip install --upgrade pip
RUN python3.12 -m pip install -r requirements.txt

COPY handler.py .

CMD [ "handler.lambda_handler" ]