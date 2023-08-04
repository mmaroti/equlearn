from typing import List
import torch

from groups import Term, VECTOR_LEN, NUM_VARS, NUM_FREE, Evaluator, Product, Inverse, Identity, Variable


class ConstantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.vector = torch.nn.Parameter(2.0 * torch.rand([VECTOR_LEN]) - 1.0)

    def forward(self):
        return self.vector


class UnaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = torch.nn.Linear(VECTOR_LEN, VECTOR_LEN)

    def forward(self, x):
        y = self.lin1(x)
        return y


class BinaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lin1 = torch.nn.Linear(VECTOR_LEN, VECTOR_LEN)
        self.lin2 = torch.nn.Linear(VECTOR_LEN, VECTOR_LEN)

    def forward(self, x, y):
        z = self.lin1(x) + self.lin2(y)
        return z


class Predictor(torch.nn.Module):
    """
    Takes a matrix of shape [NUM_NEIGHBORS, 3 * VECTOR_LEN] and
    calculates the probability distribution of shape [NUM_NEIGHBORS].
    """

    def __init__(self):
        super().__init__()

        self.lin1 = torch.nn.Linear(3 * VECTOR_LEN, 100)
        self.lin2 = torch.nn.Linear(100, 1)

    def forward(self, x):
        y = self.lin1(x)
        y = torch.relu(y)
        y = self.lin2(y)
        y = torch.squeeze(y, dim=-1)
        y = torch.softmax(y, dim=0)
        return y


class FullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.variables = [ConstantModel() for _ in range(NUM_VARS)]
        self.freeterms = [ConstantModel() for _ in range(NUM_FREE)]
        self.identity = ConstantModel()
        self.inverse = UnaryModel()
        self.product = BinaryModel()
        self.predictor = Predictor()

    def eval_variable(self, id: int):
        return self.variables[id]()

    def eval_freeterm(self, id: int):
        return self.freeterms[id]()

    def eval_identity(self):
        return self.identity()

    def eval_inverse(self, arg):
        return self.inverse(arg)

    def eval_product(self, arg0, arg1):
        return self.product(arg0, arg1)

    def parameters(self):
        params = list()
        for var in self.variables:
            params.extend(var.parameters())
        for free in self.freeterms:
            params.extend(free.parameters())
        params.extend(self.identity.parameters())
        params.extend(self.inverse.parameters())
        params.extend(self.product.parameters())
        params.extend(self.predictor.parameters())
        return params


def prepare_input(start: Term, finish: Term, evaluator: Evaluator):
    """
    Takes the start and finish terms, walks through all neighbors
    of start, and for each (start, neighbor, finish) tuple we calculate
    their vectorial embedding using eval and pack them into a matrix.
    The output shape is [NUM_NEIGHBORS, 3 * VECTOR_LEN]. It also
    returns the matching neighbor index of the next term.
    """

    start_vec = start.eval(evaluator)
    assert list(start_vec.shape) == [VECTOR_LEN]
    finish_vec = finish.eval(evaluator)
    assert finish_vec.shape == start_vec.shape

    # print(start_vec.shape)
    # print(finish_vec.shape)

    neighbors = start.all_neighbors()
    neighbors_vecs = torch.empty(
        size=(len(neighbors), VECTOR_LEN), dtype=start_vec.dtype)
    for idx, neighbor in enumerate(neighbors):
        neighbors_vecs[idx] = neighbor.eval(evaluator)

    # print(neighbors_vecs.shape)

    start_vecs = torch.unsqueeze(start_vec, dim=0).expand(len(neighbors), -1)
    finish_vecs = torch.unsqueeze(finish_vec, dim=0).expand(len(neighbors), -1)

    # print(start_vecs.shape)
    # print(finish_vecs.shape)

    input_data = torch.stack([start_vecs, neighbors_vecs, finish_vecs], dim=-1)
    # print(input_data.shape)
    input_data = input_data.reshape([len(neighbors), -1])
    # print(input_data.shape)

    return input_data, neighbors


def training(term_length=5, walk_length=3, num_steps=10000):
    assert walk_length >= 2
    full_model = FullModel()
    optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4)

    avg = 0.0
    for step in range(num_steps):
        walk = Term.random_walk(term_length, walk_length)

        # full_model.eval()
        proposed_walk = monte_carlo_search(
            full_model, walk[0], walk[-1], walk_length, 10)
        if proposed_walk:
            print(walk)
            print(proposed_walk)
            continue

        # full_model.train()
        input_data, neighbors = prepare_input(
            walk[0], walk[-1], full_model)
        next_idx = neighbors.index(walk[1])
        output_data = full_model.predictor(input_data)
        loss = 1.0 - output_data[next_idx]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg = 0.9 * avg + 0.1 * loss.item()
        if step % 100 == 0:
            print(step, avg, loss.item())
            if loss >= 0.9:
                print("start: ", walk[0], "next: ", walk[1], "end: ", walk[-1])
                sorted = torch.sort(output_data, descending=True)
                for i in range(min(10, len(neighbors))):
                    print(neighbors[sorted.indices[i].item()],
                          sorted.values[i].item())

    return full_model


def monte_carlo_search(full_model: FullModel, start: Term, end: Term, max_depth: int, num_trials: int) -> List[Term]:
    for _ in range(num_trials):
        # print("************")
        term = start
        path = []
        for _ in range(max_depth):
            path.append(term)
            if term == end:
                return path

            input_data, neighbors = prepare_input(term, end, full_model)
            output_data = full_model.predictor(input_data)
            next_idx = torch.multinomial(output_data, 1).item()
            # print(output_data)
            # print(next_idx)
            term = neighbors[next_idx]
    return None


if __name__ == '__main__':
    full_model = training(num_steps=100)

    path = monte_carlo_search(full_model,
                              Product(Variable(0), Inverse(Variable(0))),
                              Identity(),
                              4,
                              100)
    print(path)
