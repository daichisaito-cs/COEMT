ckpt=`find . -name "*.ckpt" | xargs ls -lt | awk '{ print $NF }' | grep -v '^$' | head -n 1`;python validate_jaspice_for_pfnpic2.py --model=$ckpt
