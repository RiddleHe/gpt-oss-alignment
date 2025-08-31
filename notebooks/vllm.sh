python -u \
	-m vllm.entrypoints.openai.api_server \
	--host 0.0.0.0 \
	--port 8000 \
	--model openai/gpt-oss-20b \
	--served-model-name code-gen \
	--tensor-parallel-size 8