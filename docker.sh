echo "1" | sudo -S bash -c 'setsid dockerd > /var/log/dockerd.log 2>&1 &'; sleep 4; echo "---check---"; docker info 2>&1 | head -n 20; echo "---exit:$?---"

sudo neo4j start

cd /home/fei/workspace/llama.cpp
nohup ./build/bin/llama-server \
  -m /home/fei/workspace/hub/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 40000 \
  -np 2 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --port 9000 \
  -b 2048 \
  -ub 512 \
  --chat-template-file /home/fei/workspace/hub/qwen3_nonthinking.jinja --chat-template-kwargs '{ "enableThinking": false }' \
  2>&1 | tee -a ~/logs/llama_0.8.log > /dev/null &

cd /home/fei/workspace/llama.cpp
nohup ./build/bin/llama-server \
  -m /home/fei/workspace/hub/bge-m3-Q4_K_M.gguf \
  --embedding \
  -b 4096 \
  -ub 4096 \
  -c 18192 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 9001 \
  > ~/logs/bge-m3.log 2>&1 &



cd /home/fei/workspace/knowledge_graph
python run.py 

curl -sN http://localhost:8080/v1/health