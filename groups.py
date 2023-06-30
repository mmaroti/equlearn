import random
import torch

VECTOR_LEN = 100
MODULO = 6840737
NUM_VARS = 3
NUM_FREE = 3


class Term():
    def all_neighbors(self):
        return self.eq_neighbors().union(self.sub_neighbors())

    def sub_neighbors(self):
        result = set()
        for i in self.freeterms():
            result.add(self.subst(i, Identity()))
            result.add(self.subst(i, Inverse(FreeTerm(i))))
            for j in range(NUM_VARS):
                result.add(self.subst(i, Variable(j)))
            for j in range(NUM_FREE):
                result.add(self.subst(i, Product(FreeTerm(i), FreeTerm(j))))
        return result

    @staticmethod
    def random_term(length):
        term = ''
        if length > 0:
            next = random.randint(0, 1)
            if next == 0:
                term = Inverse(Term.random_term(length-1))
            else:
                term = Product(Term.random_term((length-1)//random.randint(1, length)),
                               Term.random_term((length-1)//random.randint(1, length)))
        else:
            next = random.randint(0, NUM_VARS + 1)
            if next == 0:
                term = Identity()
            elif next == 1 and NUM_FREE > 0:
                term = FreeTerm(random.randint(0, NUM_FREE-1))
            elif NUM_VARS > 0:
                term = Variable(random.randint(0, NUM_VARS-1))
        return term

    @staticmethod
    def random_walk(term_length, walk_length):
        while True:
            next = Term.random_term(term_length)
            if not next.freeterms():
                break
        terms = [next]

        while len(terms) < walk_length:
            nexts = list(terms[-1].all_neighbors())
            next = None
            for _ in range(1 if len(terms) < walk_length//2 else 20):
                while True:
                    temp = random.choice(nexts)
                    if temp not in terms:
                        break
            if next is None or temp.length() < next.length():
                next = temp
            terms.append(next)
        return terms


class Variable(Term):
    MODEL_CS = [2.0 * torch.rand([VECTOR_LEN], requires_grad=True) - 1.0
                for _ in range(NUM_VARS)]

    def __eq__(self, other):
        return type(other) == Variable and self.id == other.id

    def __hash__(self):
        return (17 * self.id + 21) % MODULO

    def __init__(self, id):
        assert 0 <= id < NUM_VARS
        self.id = id

    def __repr__(self):
        return "xyzuvwpqrstabcdefghijklmno"[self.id]

    def eq_neighbors(self):
        result = set()
        result.add(Product(self, Identity()))
        result.add(Product(Identity(), self))
        return result

    def eval(self):
        return Variable.MODEL_CS[self.id]

    def variables(self):
        return set(self.id)

    def freeterms(self):
        return set()

    def length(self):
        return 1

    def simplify(self):
        return self

    def subst(self, id, term):
        return self


class FreeTerm(Term):
    MODEL_CS = [2.0 * torch.rand([VECTOR_LEN], requires_grad=True) - 1.0
                for _ in range(NUM_FREE)]

    def __eq__(self, other):
        return type(other) == FreeTerm and self.id == other.id

    def __hash__(self):
        return 127 + self.id

    def __init__(self, id: int):
        assert 0 <= id < NUM_FREE
        self.id = id

    def __repr__(self):
        return chr(ord("A") + self.id)

    def eq_neighbors(self):
        result = set()
        result.add(Product(self, Identity()))
        result.add(Product(Identity(), self))
        return result

    def eval(self):
        return FreeTerm.MODEL_CS[self.id]

    def variables(self):
        return set()

    def freeterms(self):
        return set([self.id])

    def length(self):
        return 1

    def simplify(self):
        return self

    def subst(self, id: int, term: Term) -> Term:
        if self.id == id:
            return term
        return self


class Identity(Term):
    MODEL_C = 2.0 * torch.rand([VECTOR_LEN], requires_grad=True) - 1.0

    def __eq__(self, other):
        return type(other) == Identity

    def __hash__(self):
        return 139

    def __init__(self):
        pass

    def __repr__(self):
        return "1"

    def eq_neighbors(self):
        result = set()
        result.add(Product(self, self))
        for id in range(NUM_FREE):
            f = FreeTerm(id)
            result.add(Product(f, Inverse(f)))
            result.add(Product(Inverse(f), f))
        return result

    def eval(self):
        return Identity.MODEL_C

    def variables(self):
        return set()

    def freeterms(self):
        return set()

    def length(self):
        return 1

    def simplify(self):
        return self

    def subst(self, id, term):
        return self


class Inverse(Term):
    MODEL_A = 2.0 * torch.rand([VECTOR_LEN, VECTOR_LEN],
                               requires_grad=True) - 1.0
    MODEL_B = 2.0 * torch.rand([VECTOR_LEN],
                               requires_grad=True) - 1.0

    def __eq__(self, other):
        return type(other) == Inverse and self.subterm == other.subterm

    def __hash__(self):
        return (19 * hash(self.subterm) + 7) % MODULO

    def __init__(self, subterm):
        self.subterm = subterm

    def __repr__(self):
        if type(self.subterm) == Variable or \
           type(self.subterm) == Identity or \
           type(self.subterm) == Inverse or \
           type(self.subterm) == FreeTerm:
            return str(self.subterm) + "\'"
        else:
            return "(" + str(self.subterm) + ")\'"

    def eq_neighbors(self):
        result = set()
        for n in self.subterm.eq_neighbors():
            result.add(Inverse(n))
        result.add(Product(self, Identity()))
        result.add(Product(Identity(), self))
        return result

    def eval(self):
        x = self.subterm.eval()
        return torch.matmul(Inverse.MODEL_A, x) + Inverse.MODEL_B

    def variables(self):
        return set(self.subterm.variables())

    def freeterms(self):
        return set(self.subterm.freeterms())

    def length(self):
        return self.subterm.length() + 1

    def simplify(self):
        subterm = self.subterm.simplify()
        if type(subterm) == Identity:
            return subterm
        elif type(subterm) == Variable:
            return Inverse(subterm)
        elif type(subterm) == FreeTerm:
            return Inverse(subterm)
        elif type(subterm) == Inverse:
            return subterm.subterm
        elif type(subterm) == Product:
            return Product(Inverse(subterm.right), Inverse(subterm.left)).simplify()
        else:
            raise ValueError()

    def subst(self, id, term):
        return Inverse(self.subterm.subst(id, term))


class Product(Term):
    MODEL_A = 2.0 * torch.rand([VECTOR_LEN, VECTOR_LEN],
                               requires_grad=True) - 1.0
    MODEL_B = 2.0 * torch.rand([VECTOR_LEN, VECTOR_LEN],
                               requires_grad=True) - 1.0
    MODEL_C = 2.0 * torch.rand([VECTOR_LEN],
                               requires_grad=True) - 1.0

    def __eq__(self, other):
        return type(other) == Product and self.left == other.left and self.right == other.right

    def __hash__(self):
        return (23 * hash(self.left) + 29 * hash(self.right) + 1) % MODULO

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        left = str(self.left)
        if type(self.left) == Product:
            left = "(" + left + ")"
        right = str(self.right)
        if type(self.right) == Product:
            right = "(" + right + ")"
        return left + "*" + right

    def eq_neighbors(self):
        result = set()
        for n in self.left.eq_neighbors():
            result.add(Product(n, self.right))
        for n in self.right.eq_neighbors():
            result.add(Product(self.left, n))
        result.add(Product(self, Identity()))
        result.add(Product(Identity(), self))
        if type(self.left) == Inverse and self.left.subterm == self.right:
            result.add(Identity())
        if type(self.right) == Inverse and self.right.subterm == self.left:
            result.add(Identity())
        if type(self.left) == Product:
            result.add(Product(self.left.left, Product(
                self.left.right, self.right)))
        if type(self.right) == Product:
            result.add(
                Product(Product(self.left, self.right.left), self.right.right))
        if type(self.left) == Identity:
            result.add(self.right)
        if type(self.right) == Identity:
            result.add(self.left)
        return result

    def eval(self):
        x = self.left.eval()
        y = self.right.eval()
        return torch.matmul(Product.MODEL_A, x) + \
            torch.matmul(Product.MODEL_B, y) + Product.MODEL_C

    def variables(self):
        return set(self.left.variables()).union(set(self.right.variables()))

    def freeterms(self):
        return set(self.left.freeterms()).union(set(self.right.freeterms()))

    def length(self):
        return self.left.length() + 1 + self.right.length()

    def simplify(self):
        left = self.left.simplify()
        right = self.right.simplify()
        if type(left) == Identity:
            return right
        elif type(right) == Identity:
            return left
        elif type(right) == Product:
            return Product(Product(left, right.left), right.right).simplify()
        elif type(left) == Inverse and left.subterm == right:
            return Identity()
        elif type(right) == Inverse and left == right.subterm:
            return Identity()
        elif type(left) == Product and type(left.right) == Inverse \
                and left.right.subterm == right:
            return left.left
        elif type(left) == Product and type(right) == Inverse \
                and left.right == right.subterm:
            return left.left
        else:
            return Product(left, right)

    def subst(self, id, term):
        return Product(self.left.subst(id, term), self.right.subst(id, term))


if __name__ == '__main__':
    print(Term.random_walk(5, 6))
