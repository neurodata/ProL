# python3 train.py -m t=1000,1500,2000,2500,3000
# python3 train.py -m process.t=1000,2000,3000 net.encoder_type=freq-fourier,sinusoid
# python3 train.py -m process.t=1000,2000,3000 net.encoder_type=sinusoid

python3 train.py -m process.t=1000,2000,3000 net.encoder_type=fourier,sinusoid net.aggregate_type=concat,sum