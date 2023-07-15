ckpt=`find . -name "*.ckpt" | xargs ls -lt | awk '{ print $NF }' | grep -v '^$' | head -n 1`;python validate_jaspice.py --model=$ckpt
