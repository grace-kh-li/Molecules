import numpy as np

class Group:
    def __init__(self, name, discrete_elements, identity):
        self.name = name
        self.discrete_elements = discrete_elements
        self.identity = identity
        self.group_name = name
        self.discrete_elements = discrete_elements
        self.identity = identity
        self.irreps = None # need to be set manually
        for i, element in enumerate(discrete_elements):
            element.group = self
            element.group_index = i

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name

class Representation:
    def __init__(self, name, group, matrices):
        self.name = name
        self.group = group
        self.representation_elements = matrices

        if matrices is None:
            self.representation_elements = [None] * len(self.group.discrete_elements)

    def _define_matrix(self, i, matrix):
        rep = RepresentationElement(self.group.discrete_elements[i], matrix)
        self.representation_elements[i] = rep

    def __str__(self):
        s = self.name + "\n"
        for e in self.representation_elements:
            s += str(e) + "\n"
        return s

    def __repr__(self):
        return str(self)

    def __getitem__(self, i):
        return self.representation_elements[i]

    def __iter__(self):
        return iter(self.representation_elements)

    def __eq__(self, other):
        out = True
        for i, e in enumerate(self.representation_elements):
            out = out and (self[i] == other[i])
        return out

    def __mul__(self, other):
        r = Representation(f"{self.name} x {other.name}", self.group, None)
        for i, g in enumerate(self.group):
            r._define_matrix(i, np.kron(self.representation_elements[i].matrix,
                                           other.representation_elements[i].matrix))
        if self.group.irreps is not None:
            for irrep in self.group.irreps:
                if irrep == r:
                    return irrep
        return r

    def __add__(self, other):
        r = Representation(f"{self.name} + {other.name}", self.group, None)
        for i, g in enumerate(self.group):
            r._define_matrix(i, direct_sum(self.representation_elements[i].matrix, other.representation_elements[i].matrix))
        return r

def direct_sum(A, B):
    """Compute the direct sum of two matrices A and B."""
    A = np.array(A)
    B = np.array(B)

    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape

    # Create a zero matrix of appropriate shape
    result = np.zeros((rows_A + rows_B, cols_A + cols_B), dtype=A.dtype)

    # Place A in the top-left block
    result[:rows_A, :cols_A] = A

    # Place B in the bottom-right block
    result[rows_A:, cols_A:] = B

    return result

class GroupElement:
    def __init__(self, name, faithful_rep, notes=""):
        self.element_name = name
        self.faithful_rep = faithful_rep
        self.group = None
        self.notes = ""
        self.group_index = 0

    def __eq__(self, other):
        """ Two group elements are equal if they have the same label, and belong to the same group """
        return np.sum(np.absolute(self.faithful_rep - other.faithful_rep)) < 1e-6

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

    def __repr__(self):
        return str(self)

    def explain(self):
        return self.notes

class FiniteGroup(Group):
    def __init__(self, name, elements, identity):
        super().__init__(name,elements, identity)
        self.irreps = ()

    def __str__(self):
        disp = self.name + "= {"
        for element in self.discrete_elements:
            disp = disp + str(element) + ", "
        disp = disp[:-2] + "}"
        return disp

    def __getitem__(self, i):
        return self.discrete_elements[i]

    def __contains__(self, element):
        if not isinstance(element, GroupElement):
            return False
        return element in self.discrete_elements

    def __iter__(self):
        return iter(self.discrete_elements)

    def multiply(self, g1, g2):
        if not isinstance(g1, GroupElement) or not isinstance(g2, GroupElement):
            raise TypeError("Cannot multiply objects that are not group elements.")
        elif g1.group != self or g2.group != self:
            raise ValueError("Cannot multiply objects that are not group elements.")
        else:
            mat = g1.faithful_rep @ g2.faithful_rep
            for e in self.discrete_elements:
                if np.sum(np.absolute(e.faithful_rep - mat)) < 1e-6:
                    return e
            print(mat)
            raise ValueError("Failed to identify the product.")

    def get_element(self, i):
        return self.discrete_elements[i]


class RepresentationElement:
    def __init__(self, element, matrix):
        self.group_element = element
        self.matrix = matrix
        self.name = element.element_name
        self.group = element.group
        self.dimension = matrix.shape

    def __mul__(self, other):
        if not isinstance(other, RepresentationElement):
            raise TypeError("Cannot multiply matrix with %s." % type(other))
        elif self.group is None or other.group is None:
            raise ValueError("Cannot operate on group elements with unknown group.")
        elif self.group != other.group:
            raise ValueError("Cannot operate on group elements with different groups.")
        else:
            m = self.matrix @ other.matrix
            k = self.group_element * other.group_element
            return RepresentationElement(k, m)

    def __eq__(self, other):
        return isinstance(other, RepresentationElement) and np.sum(np.absolute(self.matrix - other.matrix)) < 1e-8

    def __str__(self):
        return self.name + ": " + str(self.matrix)

    def __repr__(self):
        return str(self)


class FiniteGroupRepresentation(Representation):
    def __init__(self, name, group, matrices = None):
        super().__init__(name, group, matrices)
        if matrices is not None:
            if not self.check_validity():
                print("Matrices given are not a valid representation of the group. Please redefine.")
                self.representation_elements = None

    def check_validity(self):
        if self.representation_elements is None or len(self.representation_elements) != len(self.group.discrete_elements):
            print("Representation of group elements does not match group elements.")
            return False
        for m in self.representation_elements:
            if not isinstance(m, RepresentationElement):
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
                g_i = self.representation_elements[i].group_element
                g_j = self.representation_elements[j].group_element
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

    def get_matrix(self, i):
        return self.representation_elements[i].matrix


class LieAlgebraElement:
    """ Lie algebra element, defined by a matrix from a faithful representation. """
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = matrix
        self.Lie_algebra = None
        self.index = 0

    def __eq__(self, other):
        return isinstance(other, LieAlgebraElement) and np.sum(np.absolute(self.matrix - other.matrix)) < 1e-10 and self.Lie_algebra == other.Lie_algebra

    def __str__(self):
        return self.name + ": \n" + str(self.matrix)

    def __repr__(self):
        return str(self)

    def __mul__(self, c):
        g1 = LieAlgebraElement(f"{c:.2f}{self.name}", self.matrix * c)
        g1.Lie_algebra = self.Lie_algebra
        return g1

    def __add__(self, other):
        g1 = LieAlgebraElement(f"{self.name} + {other.name}", self.matrix + other.matrix)
        g1.Lie_algebra = self.Lie_algebra
        return g1

class LieAlgebra:
    def __init__(self, name, basis_elements):
        self.name = name
        self.basis_elements = basis_elements
        for i, b in enumerate(basis_elements):
            b.Lie_algebra = self
            b.index = i

    def __str__(self):
        s = self.name + "\n {"
        for e in self.basis_elements:
            s += e.name + ", "
        return s[:-2] + "}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.name == other.name and self.basis_elements == other.basis_elements

    def Lie_bracket(self, g1, g2):
        if not g1.Lie_algebra == g2.Lie_algebra == self:
            raise ValueError("Lie-algebra mismatch.")
        g3 = LieAlgebraElement(f"[{g1.name},{g2.name}]", g1.matrix * g2.matrix - g2.matrix * g1.matrix)
        g3.Lie_algebra = self

    def __getitem__(self, i):
        return self.basis_elements[i]

    def __iter__(self):
        return iter(self.basis_elements)



class LieGroup(Group):
    """ Warning: here, I define the Lie group using a Lie algebra and discrete elements.
    However, this is in general not sufficient to uniquely define a Lie group.
    Global topology information must be given to fully define the Lie group (e.g. to differentiate
    SU(2) and SO(3)). But for our purpose, this is OK. """

    def __init__(self, name,  Lie_algebra_basis, discrete_elements, identity):
        """ The matrix representation that the Lie algebra is defined by must have the same dimensionality as the discrete elements."""
        super().__init__(name, discrete_elements, identity)
        self.Lie_algebra = LieAlgebra(name, Lie_algebra_basis)

class LieAlgebraRepresentationElement:
    def __init__(self, element, matrix):
        self.algebra_element = element
        self.matrix = matrix
        self.name = element.name
        self.Lie_algebra = element.Lie_algebra
        self.dimension = matrix.shape

    def __eq__(self, other):
        return isinstance(other, LieAlgebraRepresentationElement) and self.Lie_algebra == other.Lie_algebra and np.sum(
            np.absolute(self.matrix - other.matrix)) < 1e-10

    def __str__(self):
        return self.name + ": " + str(self.matrix)

    def __repr__(self):
        return str(self)



class LieGroupRepresentation(Representation):
    def __init__(self, name, group, algebra_reps=None, discrete_reps=None):
        super().__init__(name, group, discrete_reps)
        self.algebra_rep_elements = algebra_reps
        if algebra_reps is None:
            self.algebra_rep_elements = [None] * len(self.group.Lie_algebra.basis_elements)


    def __str__(self):
        s = self.name + "\n"
        for e in self.representation_elements:
            s += str(e) + "\n"
        for e in self.algebra_rep_elements:
            s += str(e) + "\n"
        return s

    def __eq__(self, other):
        out = True
        for i, e in enumerate(self.representation_elements):
            out = out and (self[i] == other[i])
        for i, e in enumerate(self.algebra_rep_elements):
            out = out and (e == other[i])
        return out

    def __mul__(self, other):
        rep = LieGroupRepresentation(f"{self.name} x {other.name}")
        for i, r in enumerate(self.algebra_rep_elements):
            rep._define_Lie_algebra_matrix(i, np.kron(r.matrix, other.algebra_rep_elements[i].matrix))

        for i, r in enumerate(self.representation_elements):
            rep._define_matrix(i, np.kron(r.matrix, other.representation_elements[i].matrix))

        if self.group.irreps is not None:
            for irrep in self.group.irreps:
                if irrep == rep:
                    return irrep
        return rep

    def __add__(self, other):
        rep = LieGroupRepresentation(f"{self.name} x {other.name}", self.group)
        for i, r in enumerate(self.algebra_rep_elements):
            rep._define_Lie_algebra_matrix(i, direct_sum(r.matrix, other.algebra_rep_elements[i].matrix))

        for i, r in enumerate(self.representation_elements):
            rep._define_matrix(i, direct_sum(r.matrix, other.representation_elements[i].matrix))

        return rep

    def _define_Lie_algebra_matrix(self, i, mat):
        r = LieAlgebraRepresentationElement(self.group.Lie_algebra[i], mat)
        self.algebra_rep_elements[i] = r





""" Define some finite groups """
class C2vGroup(FiniteGroup):
    def __init__(self):
        e = GroupElement("e",np.array([[1,0],[0,1]]), notes="identity")
        c2 = GroupElement("C_2",np.array([[-1,0],[0,-1]]),  notes="180 degree rotation around z-axis")
        xz = GroupElement("σ_v(xz)", np.array([[1,0],[0,-1]]), notes="reflection about xz plane")
        yz = GroupElement("σ_v(yz)", np.array([[-1,0],[0,1]]), notes="reflection about yz plane")
        elements = [e, c2, xz, yz]
        super().__init__("C2v", elements, e)
        self.irreps = (C2v_A1_representation(self), C2v_A2_representation(self), C2v_B1_representation(self), C2v_B2_representation(self))


class C3vGroup(FiniteGroup):
    def __init__(self):
        e = GroupElement("e",np.array([[1,0],[0,1]]),  notes="identity")
        c3 = GroupElement("C_3", np.array([[-1/2,-np.sqrt(3)/2],[np.sqrt(3)/2,-1/2]]), notes="120 degree rotation around z-axis")
        c32 = GroupElement("(C_3)^2", np.array([[-1/2,np.sqrt(3)/2],[-np.sqrt(3)/2,-1/2]]), notes="240 degree rotation around z-axis")
        s1 = GroupElement("σ_v", np.array([[-1,0],[0,1]]), notes="reflection about plane 1")
        s2 = GroupElement("σ_v'", np.array([[1/2,np.sqrt(3)/2],[np.sqrt(3)/2,-1/2]]), notes="reflection about plane 2")
        s3 = GroupElement("σ_v''", np.array([[1/2,-np.sqrt(3)/2],[-np.sqrt(3)/2,-1/2]]), notes="reflection about plane 3")
        elements = [e, c3, c32, s1, s2, s3]
        super().__init__("C3v", elements, e)
        self.irreps = (C3v_A1_representation(self), C3v_A2_representation(self), C3v_E_representation(self))

class CsGroup(FiniteGroup):
    def __init__(self):
        e = GroupElement("e", np.array([[1]]), notes="identity")
        s = GroupElement("s", [[-1]], notes="reflection")
        elements = [e,s]
        super().__init__("Cs", elements, e)
        self.irreps = (Cs_A_prime_representation(self), Cs_A_prime_representation(self))

class CinfvGroup(LieGroup):
    def __init__(self):
        e = GroupElement("e", np.array([[1,0],[0,1]]), notes="identity")
        s = GroupElement("s", [[1,0],[0,-1]], notes="reflection")
        Lz = LieAlgebraElement("Lz",np.array([[0,-1],[1,0]]))
        super().__init__("Cinfv", [Lz], [e,s],e)

class C2v_A1_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("C2v_A1 representation", group)
        self._define_matrix(0, np.array([[1]]))
        self._define_matrix(1, np.array([[1]]))
        self._define_matrix(2, np.array([[1]]))
        self._define_matrix(3, np.array([[1]]))

class C2v_A2_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("C2v_A2 representation", group)
        self._define_matrix(0, np.array([[1]]))
        self._define_matrix(1, np.array([[1]]))
        self._define_matrix(2, np.array([[-1]]))
        self._define_matrix(3, np.array([[-1]]))

class C2v_B1_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("C2v_B1 representation", group)
        self._define_matrix(0, np.array([[1]]))
        self._define_matrix(1, np.array([[-1]]))
        self._define_matrix(2, np.array([[1]]))
        self._define_matrix(3, np.array([[-1]]))

class C2v_B2_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("C2v_B2 representation", group)
        self._define_matrix(0, np.array([[1]]))
        self._define_matrix(1, np.array([[-1]]))
        self._define_matrix(2, np.array([[-1]]))
        self._define_matrix(3, np.array([[1]]))

class C3v_A1_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("C3v_A1 representation", group)
        for i in range(6):
            self._define_matrix(i, np.array([[1]]))  # Totally symmetric

class C3v_A2_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("C3v_A2 representation", group)
        # E, C3, C3^2: +1; mirrors: -1
        signs = [1, 1, 1, -1, -1, -1]
        for i, s in enumerate(signs):
            self._define_matrix(i, np.array([[s]]))

class C3v_E_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("C3v_E representation", group)
        sqrt3 = np.sqrt(3)
        self._define_matrix(0, np.array([[1, 0], [0, 1]]))  # E
        self._define_matrix(1, np.array([[-0.5, -sqrt3 / 2], [sqrt3 / 2, -0.5]]))  # C3
        self._define_matrix(2, np.array([[-0.5, sqrt3 / 2], [-sqrt3 / 2, -0.5]]))  # C3^2
        self._define_matrix(3, np.array([[-1, 0], [0, 1]]))  # σv
        self._define_matrix(4, np.array([[0.5, sqrt3 / 2], [sqrt3 / 2, -0.5]]))   # σv′
        self._define_matrix(5, np.array([[0.5, -sqrt3 / 2], [-sqrt3 / 2, -0.5]]))  # σv″

class Cs_A_prime_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("Cs_A' representation", group)
        self._define_matrix(0, np.array([[1]]))  # E
        self._define_matrix(1, np.array([[1]]))  # σ

class Cs_A_double_prime_representation(FiniteGroupRepresentation):
    def __init__(self, group):
        super().__init__("Cs_A'' representation", group)
        self._define_matrix(0, np.array([[1]]))   # E
        self._define_matrix(1, np.array([[-1]]))  # σ

class Cinfv_Sigma_plus_representation(LieGroupRepresentation):
    def __init__(self, group):
        m_e = np.array([[1]])
        m_s = np.array([[1]])
        m_Jz = np.array([[0]])
        super().__init__("Cinfv_Sigma_plus_representation", group)
        self._define_matrix(0, m_e)
        self._define_matrix(1, m_s)
        self._define_Lie_algebra_matrix(0, m_Jz)


class Cinfv_Sigma_minus_representation(LieGroupRepresentation):
    def __init__(self, group):
        m_e = np.array([[1]])
        m_s = np.array([[-1]])
        m_Jz = np.array([[0]])
        super().__init__("Cinfv_Sigma_minus_representation", group)
        self._define_matrix(0, m_e)
        self._define_matrix(1, m_s)
        self._define_Lie_algebra_matrix(0, m_Jz)

class Cinfv_Lambda_representation(LieGroupRepresentation):
    def __init__(self, Lambda, group):
        self.Lambda = Lambda
        m_e = np.array([[1,0],[0,1]])
        m_s = np.array([[0,1],[1,0]])
        m_Jz = np.array([[-1,0],[0,1]]) * Lambda
        super().__init__(f"Cinfv_Λ={Lambda}_representation", group)
        self._define_matrix(0, m_e)
        self._define_matrix(1, m_s)
        self._define_Lie_algebra_matrix(0, m_Jz)