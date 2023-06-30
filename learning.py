import torch

from groups import Term, VECTOR_LEN

# input: start, next, end terms
# output: N * VECTOR_LEN * 3 tensor where N is the number of neighbors of start


class Predictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = torch.nn.Linear(3*VECTOR_LEN, 100)
        self.lin2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        y = self.lin1(x)
        y = torch.relu(y)
        y = self.lin2(y)
        y = torch.squeeze(y, dim=-1)
        y = torch.softmax(y, dim=0)
        return y


def prepare_input(start: Term, finish: Term):
    start_vec = start.eval()
    assert list(start_vec.shape) == [VECTOR_LEN]
    finish_vec = finish.eval()
    assert finish_vec.shape == start_vec.shape

    # print(start_vec.shape)
    # print(finish_vec.shape)

    neighbors = start.all_neighbors()
    neighbors_vecs = torch.empty(
        size=(len(neighbors), VECTOR_LEN), dtype=start_vec.dtype)
    for idx, next in enumerate(neighbors):
        neighbors_vecs[idx] = next.eval()

    # print(neighbors_vecs.shape)

    start_vecs = torch.unsqueeze(start_vec, dim=0).expand(len(neighbors), -1)
    finish_vecs = torch.unsqueeze(finish_vec, dim=0).expand(len(neighbors), -1)

    # print(start_vecs.shape)
    # print(finish_vecs.shape)

    input_data = torch.stack([start_vecs, finish_vecs, neighbors_vecs], dim=-1)
    # print(input_data.shape)
    input_data = input_data.reshape([len(neighbors), -1])
    # print(input_data.shape)

    return input_data


if __name__ == '__main__':
    model = Predictor()
    walk = Term.random_walk(5, 2)
    input_data = prepare_input(walk[0], walk[1])
    print(input_data.shape)
    output_data = model(input_data)
    print(output_data.shape)
    print(output_data)
