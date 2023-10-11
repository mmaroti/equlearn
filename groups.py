from typing import Set, List
import random

VECTOR_LEN = 100
MODULO = 6840737
NUM_VARS = 3
NUM_FREE = 3


class Evaluator():
    def eval_variable(self, id: int) -> str:
        return "v" + str(id)

    def eval_freeterm(self, id: int) -> str:
        return "f" + str(id)

    def eval_identity(self) -> str:
        return "1"

    def eval_inverse(self, arg) -> str:
        return arg + "'"

    def eval_product(self, arg0, arg1) -> str:
        return "(" + arg0 + "," + arg1 + ")"


class Term():
    def all_neighbors(self) -> List['Term']:
        return list(self.eq_neighbors().union(self.sub_neighbors()))

    def sub_neighbors(self) -> Set['Term']:
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
    def random_term(length) -> 'Term':
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
    def random_walk(term_length, walk_length) -> List['Term']:
        while True:
            next = Term.random_term(term_length)
            if not next.freeterms():
                break
        terms = [next]

        while len(terms) < walk_length:
            nexts = terms[-1].all_neighbors()
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

    def serialize(self) -> str:
        pass

    @staticmethod
    def deserialize(serialized: str) -> 'Term':
        pass

    def polishwrite(self):
        pass

class Variable(Term):
    def __eq__(self, other):
        return type(other) == Variable and self.id == other.id

    def __hash__(self):
        return (17 * self.id + 21) % MODULO

    def __init__(self, id):
        assert 0 <= id < NUM_VARS
        self.id = id

    def __repr__(self):
        return "xyzuvwpqrstabcdefghijklmno"[self.id]

    def eq_neighbors(self) -> Set[Term]:
        result = set()
        result.add(Product(self, Identity()))
        result.add(Product(Identity(), self))
        return result

    def eval(self, evaluator: Evaluator):
        return evaluator.eval_variable(self.id)

    def variables(self) -> Set[int]:
        return set(self.id)

    def freeterms(self) -> Set[str]:
        return set()

    def length(self) -> int:
        return 1

    def simplify(self) -> Term:
        return self

    def subst(self, id, term) -> Term:
        return self

    def polishwrite(self):
        return "xyzuvwpqrstabcdefghijklmno"[self.id]

class FreeTerm(Term):
    def __eq__(self, other):
        return type(other) == FreeTerm and self.id == other.id

    def __hash__(self):
        return 127 + self.id

    def __init__(self, id: int):
        assert 0 <= id < NUM_FREE
        self.id = id

    def __repr__(self):
        return chr(ord("A") + self.id)

    def eq_neighbors(self) -> Set[Term]:
        result = set()
        result.add(Product(self, Identity()))
        result.add(Product(Identity(), self))
        return result

    def eval(self, evaluator: Evaluator):
        return evaluator.eval_freeterm(self.id)

    def variables(self) -> Set[int]:
        return set()

    def freeterms(self) -> Set[int]:
        return set([self.id])

    def length(self) -> int:
        return 1

    def simplify(self) -> Term:
        return self

    def subst(self, id: int, term: Term) -> Term:
        if self.id == id:
            return term
        return self

    def polishwrite(self):
        return chr(ord("A") + self.id)

class Identity(Term):
    def __eq__(self, other):
        return type(other) == Identity

    def __hash__(self):
        return 139

    def __init__(self):
        pass

    def __repr__(self):
        return "1"

    def eq_neighbors(self) -> Set[Term]:
        result = set()
        result.add(Product(self, self))
        for id in range(NUM_FREE):
            f = FreeTerm(id)
            result.add(Product(f, Inverse(f)))
            result.add(Product(Inverse(f), f))
        return result

    def eval(self, evaluator: Evaluator):
        return evaluator.eval_identity()

    def variables(self) -> Set[int]:
        return set()

    def freeterms(self) -> Set[int]:
        return set()

    def length(self) -> int:
        return 1

    def simplify(self) -> Term:
        return self

    def subst(self, id, term) -> Term:
        return self

    def polishwrite(self):
        return "1"

class Inverse(Term):
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

    def eq_neighbors(self) -> Set[Term]:
        result = set()
        for n in self.subterm.eq_neighbors():
            result.add(Inverse(n))
        result.add(Product(self, Identity()))
        result.add(Product(Identity(), self))
        return result

    def eval(self, evaluator: Evaluator):
        x = self.subterm.eval(evaluator)
        return evaluator.eval_inverse(x)

    def variables(self) -> Set[int]:
        return set(self.subterm.variables())

    def freeterms(self) -> Set[int]:
        return set(self.subterm.freeterms())

    def length(self) -> int:
        return self.subterm.length() + 1

    def simplify(self) -> Term:
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

    def subst(self, id, term) -> Term:
        return Inverse(self.subterm.subst(id, term))

    def polishwrite(self):
        return "-" + self.subterm.polishwrite()

class Product(Term):
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

    def eq_neighbors(self) -> Set[Term]:
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

    def eval(self, evaluator: Evaluator):
        x = self.left.eval(evaluator)
        y = self.right.eval(evaluator)
        return evaluator.eval_product(x, y)

    def variables(self) -> Set[int]:
        return set(self.left.variables()).union(set(self.right.variables()))

    def freeterms(self) -> Set[int]:
        return set(self.left.freeterms()).union(set(self.right.freeterms()))

    def length(self) -> int:
        return self.left.length() + 1 + self.right.length()

    def simplify(self) -> Term:
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

    def subst(self, id, term) -> Term:
        return Product(self.left.subst(id, term), self.right.subst(id, term))

    def polishwrite(self):
        return "*" + self.left.polishwrite() + self.right.polishwrite()

if __name__ == '__main__':
    walk = Term.random_walk(5, 6)
    print(walk)
    term = walk[0]
    evaluator = Evaluator()
    print(term.eval(evaluator))
    print(Product(Product(Inverse(Product(Inverse(Inverse(Variable(0))), Variable(1))), Inverse(FreeTerm(0))), Identity()))
    print(Product(Product(Inverse(Product(Inverse(Inverse(Variable(0))), Variable(1))), Inverse(FreeTerm(0))), Identity()).polishwrite())