import os
import pandas as pd
import torch
from train import train
from test import test

def main(model_name, parameters_name, save_name, cuda_device=0):

	train_loss_list = []
	valid_loss_list = []
	train_accuracy_list = []
	valid_accuracy_list = []
	test_accuracy_list = []

	base_path = os.path.dirname(os.path.abspath(__file__))
	parameters = pd.read_csv(os.path.join(base_path, parameters_name), header=0)
	device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

	for index in range(parameters.shape[0]):

		batch_size = int(parameters.iloc[index]['batch_size'])
		epochs = int(parameters.iloc[index]['epochs'])
		learning_rate = parameters.iloc[index]['learning_rate']

		state_name = f"{batch_size}_{epochs}_{learning_rate}"
		save_path = os.path.join(base_path, save_name, state_name)

		if not os.path.exists(save_path):
			os.mkdir(save_path)

		(
			train_loss,
			valid_loss,
			train_accuracy,
			valid_accuracy
		) = train(
			model_name=model_name,
			batch_size=batch_size,
			epochs=epochs,
			learning_rate=learning_rate,
			device=device,
			base_path=base_path,
			save_path=save_path
		)

		test_accuracy = test(
			model_name=model_name,
			device=device,
			base_path=base_path,
			save_path=save_path
		)

		train_loss_list.append(train_loss)
		valid_loss_list.append(valid_loss)
		train_accuracy_list.append(train_accuracy)
		valid_accuracy_list.append(valid_accuracy)
		test_accuracy_list.append(test_accuracy)

	parameters['train_loss'] = pd.Series(train_loss_list)
	parameters['valid_loss'] = pd.Series(valid_loss_list)
	parameters['train_accuracy'] = pd.Series(train_accuracy_list)
	parameters['valid_accuracy'] = pd.Series(valid_accuracy_list)
	parameters['test_accuracy'] = pd.Series(test_accuracy_list)

	parameters.to_csv(os.path.join(base_path, f'{save_name}.csv'), na_rep='NAN', index=False, float_format='%.4f')

if __name__ == '__main__':
	# part 1
	main(model_name="ExampleCNN", parameters_name='parameters.csv', save_name="part_1_result", cuda_device=0)

	# part 2
	main(model_name="MyCNN", parameters_name='parameters.csv', save_name="part_2_result", cuda_device=0)
