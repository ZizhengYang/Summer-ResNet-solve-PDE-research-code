# ===========
# padding method
# ===========

# padding and zero padding
def padding(original, starting_padding, end_padding):
    return np.hstack((starting_padding, original, end_padding)).tolist()

def zero_padding(original, num):
    zero_list = [0 for i in range(num)]
    return padding(original, zero_list, zero_list)

def border_padding(original, num):
    starting = original[0]
    ending = original[len(original)-1]
    starting_padding = [starting for i in range(num)]
    end_padding = [ending for i in range(num)]
    return padding(original, starting_padding, end_padding)

def recursive_padding(original, num):
    starting_padding = original[-num:]
    end_padding = origin[:num]
    return padding(original, starting_padding, end_padding)

def random_padding(original, num):
    max_value = np.max(original)
    min_value = np.min(original)
    random_list_1 = [random.randint(min_value, max_value) for i in range(num)]
    random_list_2 = [random.randint(min_value, max_value) for i in range(num)]
    return padding(original, random_list_1, random_list_2)