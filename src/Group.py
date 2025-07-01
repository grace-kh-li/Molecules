class GroupElement:
    def __init__(self, name):
        self.element_name = name
        self.group = None
        self.group_index = 0

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

class FiniteGroup:
    def __init__(self, name, elements, identity, operation_table):
        self.group_name = name
        self.elements = elements
        self.identity = identity
        self.operation_table = operation_table
        for i, element in enumerate(elements):
            element.group = self
            element.group_index = i

    def __str__(self):
        return self.group_name

    def __eq__(self, other):
        return isinstance(other, FiniteGroup) and self.group_name == other.group_name

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

