# ==============
# FullconnectedResNet
# ==============

class FullconnectedResNet(nn.Module):
    def __init__(self, i, o, layer_data):
        super(FullconnectedResNet, self).__init__()
        self.layers = torch.nn.Sequential()
        self.layers.add_module("linear_1", nn.Linear(i, layer_data[0]))
        self.layers.add_module("relu_1", nn.ReLU())
        for index in range(len(layer_data)-1):
            self.layers.add_module("linear_"+str(index+2), nn.Linear(layer_data[index], layer_data[index+1]))
            self.layers.add_module("relu_"+str(index+2), nn.ReLU())
        self.layers.add_module("linear_"+str(len(layer_data)+1), nn.Linear(layer_data[len(layer_data)-1], o))
        self.layers.add_module("relu_"+str(len(layer_data)+1), nn.ReLU())
        
    def forward(self, x):
        output = self.layers(x)
        return output + torch.mean(x)

    def load_model(self, save_path):
        self.load_state_dict(torch.load(save_path))

    def save_model(self, save_path):
        torch.save(self.state_dict(), save_path)

# ==============
# Training
# ==============

def training_the_model_FullconnectedResNet(delta_x=1/20 * np.pi, delta_t=1/20 * np.pi, xmin=0, tmin=0, xmax=2 * np.pi, tmax=2 * np.pi, analytical_eq=PDE_analytical_solu, gen_solution=gen_cell_average_solu,
                       allTheTime=False,
                       num=3, padding=border_padding,
                       i=7, o=1, layer_data=[6, 6],
                       lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,
                       iteration=2,
                       json_file="config.json", solu_file="solu.csv", input_file="inputs.txt", loss_file="train_losses.txt", outout_file="outputs.txt", model_file = "model"):
    # ===============
    # Prepare the training set
    # ===============
    x, t, solu = gen_solution(delta_x=delta_x, delta_t=delta_t, xmin=xmin, tmin=tmin, xmax=xmax, tmax=tmax, analytical_eq=PDE_analytical_solu)
    # x0, t0, solu0 = gen_solution(t_index=0, delta_x=delta_x, delta_t=delta_t, xmin=xmin, tmin=tmin, xmax=xmax, tmax=tmax, analytical_eq=PDE_analytical_solu)
    # x1, t1, solu1 = gen_solution(t_index=1, delta_x=delta_x, delta_t=delta_t, xmin=xmin, tmin=tmin, xmax=xmax, tmax=tmax, analytical_eq=PDE_analytical_solu)
    actual_solu = solu
    if not allTheTime:
        solu = solu[:2]
    pairs = get_trainingset_all(solu, num=num, padding=padding)
    # print(pairs)
    # ==============
    # Set the saving pathes
    # ==============
    f = open("counter.txt")
    list_of_counters = []
    for line in open("counter.txt"):
        list_of_counters.append(line)
    experiment_counter = int(list_of_counters[0])
    folder_name = "experiment-" + list_of_counters[0]
    experiment_counter += 1
    with open("counter.txt","w") as f:
        f.write(str(experiment_counter))
    #     folder_name = "ResNet" \
    #     +" "+"delta_x="+str(round(delta_x, 3))+" "+"delta_t="+str(round(delta_t, 3))+" " \
    #     +"xmin="+str(round(xmin, 3))+" "+"tmin="+str(round(tmin, 3))+" "+"xmax="+str(round(xmax, 3))+" "+"tmax="+str(round(tmax, 3)) \
    #     +" "+"analytical_eq="+analytical_eq.__name__+" "+"gen_solution="+gen_solution.__name__ \
    #     +" "+"input_dim="+str(i)+" "+"layer_data="+str(layer_data) \
    #     +" "+"iteration="+str(iteration)
    folder = os.path.exists(folder_name)
    if not folder:
        os.makedirs(folder_name)
    json_file=folder_name+"/"+json_file
    solu_file_used=folder_name+"/"+"u_ "+solu_file
    solu_file_actual=folder_name+"/"+"a_ "+solu_file
    input_file=folder_name+"/"+input_file
    loss_file=folder_name+"/"+loss_file
    outout_file=folder_name+"/"+outout_file
    model_file=folder_name+"/"+model_file
    # =================
    # Set up model & optimizer
    # =================
    model = FullconnectedResNet(i=i, o=o, layer_data=layer_data)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    model.zero_grad()
    optimizer.zero_grad()
    criterion = nn.MSELoss()
    # ===========
    # Train the model
    # ===========
    model.train()
    list_of_loss = []
    list_of_output = []
    counter = 0
    for itera in range(iteration):
        for pair in pairs:
            output = model(torch.FloatTensor(pair["train"]))
            loss = criterion(output, torch.FloatTensor([pair["target"]]))
            loss.backward()
            optimizer.step()
            list_of_loss.append(loss.item())
            list_of_output.append(output)
            print("="*20)
            print("The", counter, "time of training")
            print("We are using:", pair["train"])
            print("It should have be:", pair["target"])
            print("But prediction is:", output)
            print("The training loss is:", loss.item())
            counter += 1
    # ============
    # Save the json data
    # ============
    data_setting = {'type': 'data', 'padding': str(border_padding), 'data_getter': str(get_trainingset_all)}
    range_setting = {'type': 'range', 'delta_x': delta_x,'delta_t': delta_t,'xmin': xmin, 'tmin': tmin, 'xmax': xmax, 'tmax': tmax}
    optimizer_setting = {'type': 'optimizer', 'optimizer': str(optim.Adam), 'learning_rate': lr, 'betas_values': betas, 'eps': eps, 'weight_decay': weight_decay, 'amsgrad': amsgrad}
    model_setting = {'type': 'model', 'model': 'full connected ResNet', 'input_dimension': i, 'output_dimension': o, 'hidden_dimensions': layer_data}
    differential_eq_setting = {'type': 'equation', 'analytical_eq': str(analytical_eq), 'range_setting': range_setting}
    training_setting = {'type': 'training', 'iteration': iteration, 'range_setting': range_setting, 'model_setting': model_setting, 'optimizer_setting': optimizer_setting}
    json_data = {'differential_eq_setting': differential_eq_setting, 'training_setting': training_setting}
    save_json(json_file, json_data)
    # ========================
    # Save the calculated analytical solution
    # ========================
    save_csv(solu_file_used, solu)
    save_csv(solu_file_actual, actual_solu)
    # ====================
    # Save the inputs/outputs/losses
    # ====================
    save_list(input_file, pairs)
    save_list(outout_file, list_of_output)
    save_list(loss_file, list_of_loss)
    # ==========
    # Save the model
    # ==========
    model.save_model(model_file)

# ==============
# Evaluation
# ==============

def testing_the_model_FullconnectedResNet(delta_x=1/20 * np.pi, delta_t=1/20 * np.pi, xmin=0, tmin=0, xmax=2 * np.pi, tmax=2 * np.pi, analytical_eq=PDE_analytical_solu, gen_solution=gen_cell_average_solu,
                       num=3, padding=border_padding,
                       i=7, o=1, layer_data=[6, 6],
                       experiment_id = 1,
                       model_file = "model", loss_file="eval_losses.txt", err_file="errs.txt", predict_file="prediction.csv"):
    # ===============
    # Prepare the training set
    # ===============
    x, t, solu = gen_solution(delta_x=delta_x, delta_t=delta_t, xmin=xmin, tmin=tmin, xmax=xmax, tmax=tmax, analytical_eq=PDE_analytical_solu)
    pairs = get_testingset_all(solu, num=num, padding=padding)
    # ==============
    # Set the saving pathes
    # ==============
    folder_name = "experiment-" + str(experiment_id)
    model_file=folder_name+"/"+model_file
    loss_file=folder_name+"/"+loss_file
    err_file=folder_name+"/"+err_file
    predict_file=folder_name+"/"+predict_file
    # =================
    # Set up model & optimizer
    # =================
    model = FullconnectedResNet(i=i, o=o, layer_data=layer_data)
    model.load_model(model_file)
    criterion = nn.MSELoss()
    # ===========
    # Train the model
    # ===========
    model.eval()
    list_of_loss = []
    list_of_error = []
    model_result = []
    counter = 0
    for j in range(len(pairs)-1-1):
        pairs_t = pairs[j]
        model_result_t = []
        for pair in pairs_t:
            output = model(torch.FloatTensor(pair["train"]))
            loss = criterion(output, torch.FloatTensor([pair["target"]]))
            error = str(100 * np.absolute(output.item() - pair["target"]) / pair["target"])+"%"
            print("="*20)
            print("The", counter, "time of training")
            print("The time segement is:" j)
            print("The input pair is:", pair["train"])
            print("It should have be:", pair["target"])
            print("The evaluation loss is:", loss)
            print("The error in percentage is:", error)
            model_result_t.append(output.item())
            list_of_loss.append(loss.item())
            list_of_error.append(error)
            counter += 1
        model_result.append(model_result_t)
    # =================
    # Save the losses and error
    # =================
    save_list(loss_file, list_of_loss)
    save_list(err_file, list_of_error)
    # =============
    # Save the prediction
    # =============
    save_csv(predict_file, model_result)