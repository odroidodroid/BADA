python main.py --multirun rng.seed=range(100) data=two_bells/0.1k model=gaussian_process trainer=gaussian_process experiment_name=two_bells_0.1k acquisition.objective=bald
python main.py --multirun rng.seed=range(100) data=two_bells/1k model=gaussian_process trainer=gaussian_process experiment_name=two_bells_1k acquisition.objective=bald
python main.py --multirun rng.seed=range(100) data=two_bells/10k model=gaussian_process trainer=gaussian_process experiment_name=two_bells_10k acquisition.objective=bald
python main.py --multirun rng.seed=range(100) data=two_bells/100k model=gaussian_process trainer=gaussian_process experiment_name=two_bells_100k acquisition.objective=bald
python main.py --multirun rng.seed=range(100) data=two_bells/100k model=gaussian_process trainer=gaussian_process experiment_name=two_bells_100k acquisition.objective=random
python main.py --multirun rng.seed=range(100) data=two_bells/100k model=gaussian_process trainer=gaussian_process experiment_name=two_bells_100k acquisition.objective=epig