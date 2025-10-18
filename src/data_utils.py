import torch
from bidict import bidict
import random 

"""
1. noise
2. EOS
3. how to test b -> c
    - train on a subset of b -> c
    - probing
4. Model    
    - Multi-head
    - Multi-layer
"""

class ImplicitReasoningTask:
    def __init__(self, num_name, num_city, num_animal, num_signal = 2, num_noise = 6, seed = 42, 
        num_probing = 0,
        # even_distribution = False, 
        use_eos = True,
        train_test_split = 0.75,
        random_draw = False
        ):
        random.seed(seed)
        # Vocabulary: num_signal + num_name + num_city * 2  (num_city + num_animal) + num_nois + EOS

        self.vocab_size = num_signal + num_name + num_city + num_animal + num_noise + num_probing + use_eos
        # if even_distribution:
        #     assert num_name % num_city == 0 # force the name to be evenly distributed over cities
        # initialize the data
        self.signal_list = [i for i in range(num_signal)]
        self.city_list = [i for i in range(num_signal, num_signal + num_city)]
        self.animal_list = [i for i in range(num_signal + num_city, num_signal + num_city + num_animal)]
        self.probing_list = [i for i in range(num_signal + num_city + num_animal, num_signal + num_city + num_animal + num_probing)]
        offset = num_signal + num_city + num_animal + num_probing
        self.num_signal = num_signal
        self.name_list = []
        for i in range(offset, offset + num_name):
            self.name_list.append(i)
        if not random_draw:
            self.name_city_map = {self.name_list[i]: self.city_list[i % num_city] for i in range(num_name)}
            self.city_animal_map = {self.city_list[i]: self.animal_list[i % num_animal] for i in range(num_city)}
            if num_probing > 0:
                self.city_probing_map = {self.city_list[i]: self.probing_list[i % num_probing] for i in range(num_city)}
        else:
            random.shuffle(self.city_list)
            random.shuffle(self.animal_list)
            if num_probing > 0:
                random.shuffle(self.probing_list)
                self.city_probing_map = {self.city_list[i]: self.probing_list[i % num_probing] for i in range(num_city)}
            self.name_city_map = {self.name_list[i]: self.city_list[i % num_city] for i in range(num_name)}
            self.city_animal_map = {self.city_list[i]: self.animal_list[i % num_animal] for i in range(num_city)}
        
        self.name_animal_map = {self.name_list[i]: self.city_animal_map[self.name_city_map[self.name_list[i]]] for i in range(num_name)}
            # self.name_city_map[self.name_list[i]] = self.city_list[i % num_city]
            # self.name_animal_map[self.name_list[i]] = self.animal_list[i % num_city]
        # self.train_name_list = self.name_list[:num_name * 2 // 4]
        self.train_name_list = self.name_list[:int(num_name * train_test_split)]
        self.test_name_list = self.name_list[int(num_name * train_test_split):]
        # self.city_list = [i for i in range(num_signal + num_name, num_signal + num_name + num_city)]
        if num_signal > 2:
            self.train_city_list = self.city_list[:num_city // 2]
            self.test_city_list = self.city_list[num_city // 2:]
        
        self.noise_list = [i for i in range(offset + num_name, self.vocab_size - use_eos)]
        if use_eos:
            self.EOS_token = self.vocab_size - 1
        else:
            self.EOS_token = None
        
        # perturb the animal_list
        # random.shuffle(self.animal_list)
        print("animal list: ", self.animal_list)
        print('city list: ', self.city_list)
        print('noise list: ', self.noise_list)
        if num_probing > 0:
            print('probing list: ', self.probing_list)
        # city_animal_ma


        print("city animal map: ", self.city_animal_map)
        # city_candidate_list = self.city_list * (num_name // num_city)
        # self.city_train_list = city_candidate_list[:num_name * 3 // 4]
        # self.city_test_list = city_candidate_list[num_name * 3 // 4:]

        # random.shuffle(self.city_train_list)
        # random.shuffle(self.city_test_list)

        # name_city_train_map = {self.train_name_list[i]: self.city_train_list[i] for i in range(len(self.train_name_list))}
        # name_city_test_map = {self.test_name_list[i]: self.city_test_list[i] for i in range(len(self.test_name_list))}
        # print("name city train map: ", name_city_train_map)
        # print("name city test map: ", name_city_test_map)
        # self.name_city_map = {**name_city_train_map, **name_city_test_map}
        print("name city map: ", self.name_city_map)
        # self.name_animal_map = {self.name_list[i]: self.city_animal_map[self.name_city_map[self.name_list[i]]] for i in range(num_name)}
        print("name animal map: ", self.name_animal_map)
    def get_batch(self,
        batch_size:int, 
        seq_len:int,
        get_train:bool = True,
        task: str = 'fact',
    ):
        min_seq_len = 2 + (self.EOS_token is not None)
        assert seq_len >= min_seq_len, f"seq_len should be at least {min_seq_len}"

        if get_train:
            name_list = self.train_name_list
        else:
            name_list = self.test_name_list
        input_tensors = []
        output_tensors = []
        for i in range(batch_size):
            if task == 'fact':
                name = random.choice(name_list)
                label = self.name_city_map[name]
                signal = self.signal_list[0]
            elif task == 'impl':
                name = random.choice(name_list)
                label = self.name_animal_map[name]
                signal = self.signal_list[1]
            elif task == 'self-ref': # self-reference
                assert self.num_signal > 2, "self-ref task requires at least 3 signals"
                name = random.choice(name_list)
                label = self.noise_list[0]
                signal = self.signal_list[2]
            elif task == 'probing':
                assert self.num_signal > 2, "probing task requires at least 3 signals"
                if get_train:
                    name = random.choice(self.city_list)
                    label = self.city_probing_map[name]
                else:
                    name = random.choice(self.name_list)
                    label = self.city_probing_map[self.name_city_map[name]]
                signal = self.signal_list[2]
            input_strings = random.choices(self.noise_list, k = seq_len - min_seq_len)
            name_idx = random.randint(0, len(input_strings) + 1)
            input_strings.insert(name_idx, name)
            signal_idx = random.randint(name_idx + 1, len(input_strings) + 1)
            input_strings.insert(signal_idx, signal)

            if self.EOS_token is not None:
                input_strings.append(self.EOS_token)
            
            name_pos = input_strings.index(name)
            signal_pos = input_strings.index(signal)
            assert name_pos < signal_pos
            assert len(input_strings) == seq_len
            input_tensors.append(input_strings)
            output_tensors.append(label)
        # return input_tensors, output_tensors
        return torch.tensor(input_tensors), torch.tensor(output_tensors)