ckpt=`find . -name "*.ckpt" | xargs ls -lt | cut -f 10  -d ' ' | head -n 1`;python validate_jaspice.py --model=$ckpt
