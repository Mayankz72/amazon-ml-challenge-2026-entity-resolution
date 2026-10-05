# common settings; heavy files on node-local /tmp
A=${AMZML_HOME:-$HOME/amzml}
W=$A/work
T=${TMPDIR:-/tmp}/amzml
PY=$A/env/bin/python
export BER_DATA_DIR=$A/dataset BER_DENSE_SPLIT=1 BER_VAL_STATES=il,ka TOKENIZERS_PARALLELISM=false
V6_STATES=va,tn,kl,mh,dl,tx,ny,nc,up
# GPU sharing on the V100 node is not enforced: refuse to start on a GPU that someone else already fills
gpucheck() { $PY -c "import torch,sys;f,t=torch.cuda.mem_get_info();print('gpu free GB',round(f/1e9,1),flush=True);sys.exit(0 if f>20e9 else 75)"; }
