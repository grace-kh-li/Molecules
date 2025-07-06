import numpy as np


class GroupElement:
    def __init__(self, name, notes=""):
        self.element_name = name
        self.group = None
        self.group_index = 0
        self.notes = ""

    def __eq__(self, other):
        """ Two group elements are equal if they have the same label, and belong to the same group """
        return isinstance(other, GroupElement) and self.element_name == other.element_name and self.group == other.group

    def __mul__(self, other):
        """ g1(self) * g2(other) """
        if not isinstance(other, GroupElement):
            raise TypeError("Cannot multiply group element with %s." % type(other))
        elif self.group is None or other.group is None:
            raise ValueError("Cannot operate on group elements with unknown group.")
        elif self.group != other.group:
            raise ValueError("Cannot operate on group elements with different groups.")
        else:
            return self.group.multiply(self, other)

    def __str__(self):
        return self.element_name

    def explain(self):
        return self.notes

class AbstractFiniteGroup:
    def __init__(self, name, elements, identity, operation_table):
        self.group_name = name
        self.elements = elements
        self.identity = identity
        self.operation_table = operation_table
        for i, element in enumerate(elements):
            element.group = self
            element.group_index = i

    def __str__(self):
        disp = self.group_name + "= {"
        for element in self.elements:
            disp = disp + str(element) + ", "
        disp = disp[:-2] + "}"
        return disp

    def __eq__(self, other):
        return isinstance(other, AbstractFiniteGroup) and self.group_name == other.group_name

    def __contains__(self, element):
        if not isinstance(element, GroupElement):
            return False
        return element in self.elements

    def multiply(self, g1, g2):
        if not isinstance(g1, GroupElement) or not isinstance(g2, GroupElement):
            raise TypeError("Cannot multiply objects that are not group elements.")
        elif g1.group != self or g2.group != self:
            raise ValueError("Cannot multiply objects that are not group elements.")
        else:
            i_mul = self.operation_table[g1.group_index, g2.group_index]
            return self.elements[i_mul]

    def get_element(self, i):
        return self.elements[i]


class GroupRepresentationElement:
    def __init__(self, element, matrix):
        self.group_element = element
        self.matrix = matrix
        self.name = element.element_name
        self.group = element.group
        self.dimension = matrix.shape

    def __mul__(self, other):
        if not isinstance(other, GroupRepresentationElement):
            raise TypeError("Cannot multiply matrix with %s." % type(other))
        elif self.group is None or other.group is None:
            raise ValueError("Cannot operate on group elements with unknown group.")
        elif self.group != other.group:
            raise ValueError("Cannot operate on group elements with different groups.")
        else:
            m = self.matrix @ other.matrix
            k = self.group_element * other.group_element
            return GroupRepresentationElement(k, m)

    def __eq__(self, other):
        return isinstance(other, GroupRepresentationElement) and self.group == other.group and np.sum(np.absolute(self.matrix - other.matrix)) < 1e-10

    def __str__(self):
        return self.name + ": " + str(self.matrix)


class GroupRepresentation:
    def __init__(self, name, group, matrices = None):
        self.name = name
        self.group = group
        self.representation_elements = matrices
        if matrices is not None:
            if not self.check_validity():
                print("Matrices given are not a valid representation of the group. Please redefine.")
                self.representation_elements = None
        if matrices is None:
            self.representation_elements = [None] * len(self.group.elements)

    def __str__(self):
        s = ""
        for e in self.representation_elements:
            s += str(e) + "\n"
        return s



    def check_validity(self):
        if self.representation_elements is None or len(self.representation_elements) != len(self.group.elements):
            print("Representation of group elements does not match group elements.")
            return False
        for m in self.representation_elements:
            if not isinstance(m, GroupRepresentationElement):
                print("Incomplete definition.")
                return False

        dim = self.representation_elements[0].dimension
        for m in self.representation_elements:
            if m.dimension != dim:
                print("Matrices must have same dimension.")
                return False

        # check that multiplication table of the group is satisfied
        for i in range(len(self.representation_elements)):
            for j in range(len(self.representation_elements)):
                g_i = self.group.get_element(i)
                g_j = self.group.get_element(j)
                g_ij = g_i * g_j
                k = g_ij.group_index
                rep_k = self.representation_elements[k]

                if rep_k != self.representation_elements[i] * self.representation_elements[j]:
                    print(g_i)
                    print(g_j)
                    print("supposed to be:")
                    print(rep_k)
                    print("got:")
                    print(self.representation_elements[i] * self.representation_elements[j])
                    return False

        return True


    def define_matrix(self, i, matrix):
        rep = GroupRepresentationElement(self.group.elements[i], matrix)
        self.representation_elements[i] = rep

    def get_matrix(self, i):
        return self.representation_elements[i].matrix



""" Define some finite groups """
class C2vGroupAbstract(AbstractFiniteGroup):
    def __init__(self):
        e = GroupElement("e", notes="identity")
        c2 = GroupElement("C_2", notes="180 degree rotation around z-axis")
        xz = GroupElement("σ_v(xz)", notes="reflection about xz plane")
        yz = GroupElement("σ_v(yz)", notes="reflection about yz plane")
        elements = [e, c2, xz, yz]
        table = np.array([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]])
        super().__init__("C2v", elements, e, table)


class C3vGroupAbstract(AbstractFiniteGroup):
    def __init__(self):
        e = GroupElement("e", notes="identity")
        c3 = GroupElement("C_3", notes="120 degree rotation around z-axis")
        c32 = GroupElement("(C_3)^2", notes="240 degree rotation around z-axis")
        s1 = GroupElement("σ_v", notes="reflection about plane 1")
        s2 = GroupElement("σ_v'", notes="reflection about plane 2")
        s3 = GroupElement("σ_v''", notes="reflection about plane 3")
        elements = [e, c3, c32, s1, s2, s3]
        c3v_table = np.array([
            [0, 1, 2, 3, 4, 5],  # E
            [1, 2, 0, 4, 5, 3],  # C3
            [2, 0, 1,5, 3, 4],  # C3^2
            [3, 5, 4, 0, 2, 1],  # σv
            [4, 3, 5, 1, 0, 2],  # σv′
            [5, 4, 3, 2, 1, 0]  # σv″
        ])
        super().__init__("C3v", elements, e, c3v_table)

class CsGroupAbstract(AbstractFiniteGroup):
    def __init__(self):
        e = GroupElement("e", notes="identity")
        s = GroupElement("s", notes="reflection")
        elements = [e,s]
        cs_table = np.array([
            [0, 1],  # E
            [1, 0]  # σ
        ])
        super().__init__("Cs", elements, e, cs_table)


class C2v_A1_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("C2v_A1", C2vGroupAbstract())
        self.define_matrix(0, np.array([[1]]))
        self.define_matrix(1, np.array([[1]]))
        self.define_matrix(2, np.array([[1]]))
        self.define_matrix(3, np.array([[1]]))

class C2v_A2_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("C2v_A1", C2vGroupAbstract())
        self.define_matrix(0, np.array([[1]]))
        self.define_matrix(1, np.array([[1]]))
        self.define_matrix(2, np.array([[-1]]))
        self.define_matrix(3, np.array([[-1]]))

class C2v_B1_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("C2v_B1", C2vGroupAbstract())
        self.define_matrix(0, np.array([[1]]))
        self.define_matrix(1, np.array([[-1]]))
        self.define_matrix(2, np.array([[1]]))
        self.define_matrix(3, np.array([[-1]]))

class C2v_B2_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("C2v_B2", C2vGroupAbstract())
        self.define_matrix(0, np.array([[1]]))
        self.define_matrix(1, np.array([[-1]]))
        self.define_matrix(2, np.array([[-1]]))
        self.define_matrix(3, np.array([[1]]))

class C3v_A1_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("C3v_A1", C3vGroupAbstract())
        for i in range(6):
            self.define_matrix(i, np.array([[1]]))  # Totally symmetric

class C3v_A2_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("C3v_A2", C3vGroupAbstract())
        # E, C3, C3^2: +1; mirrors: -1
        signs = [1, 1, 1, -1, -1, -1]
        for i, s in enumerate(signs):
            self.define_matrix(i, np.array([[s]]))

class C3v_E_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("C3v_E", C3vGroupAbstract())
        sqrt3 = np.sqrt(3)
        self.define_matrix(0, np.array([[1, 0], [0, 1]]))  # E
        self.define_matrix(1, np.array([[-0.5, -sqrt3/2], [sqrt3/2, -0.5]]))  # C3
        self.define_matrix(2, np.array([[-0.5, sqrt3/2], [-sqrt3/2, -0.5]]))  # C3^2
        self.define_matrix(3, np.array([[1, 0], [0, -1]]))  # σv
        self.define_matrix(4, np.array([[-0.5, sqrt3/2], [sqrt3/2, 0.5]]))   # σv′
        self.define_matrix(5, np.array([[-0.5, -sqrt3/2], [-sqrt3/2, 0.5]]))  # σv″

class Cs_A_prime_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("Cs_A'", CsGroupAbstract())
        self.define_matrix(0, np.array([[1]]))  # E
        self.define_matrix(1, np.array([[1]]))  # σ

class Cs_A_double_prime_representation(GroupRepresentation):
    def __init__(self):
        super().__init__("Cs_A''", CsGroupAbstract())
        self.define_matrix(0, np.array([[1]]))   # E
        self.define_matrix(1, np.array([[-1]]))  # σ
