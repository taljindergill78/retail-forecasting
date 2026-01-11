FROM public.ecr.aws/lambda/python:3.11

# psycopg2-binary is easiest for Lambda container
RUN pip install --no-cache-dir psycopg2-binary boto3

COPY app.py ${LAMBDA_TASK_ROOT}/app.py

CMD ["app.lambda_handler"]
