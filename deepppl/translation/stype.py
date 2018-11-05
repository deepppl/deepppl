""" 
 Copyright 2018 IBM Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from .exceptions import IncompatibleTypes

class TypeDesc(object):
    """ This is the (abstract) base class for type descriptions.
        These are wrapped by Type_ objects, which are generally what you want to use.
    """
    pass

# This defines a type system for Stan, extended with support for Type Variables
# We do not currently model constraints.

class Type_(object):
    """ Creates a new type object, which represents the type of something.
        It contains a type description, describing the structure of the type.
        This type description may be shared with other Type_ objects, indicating 
        that the two types are the same.
    """
    def __init__(self, desc:TypeDesc):
        """ Creates a new type, wrapping a given type description (representation).
            Generally, the classmethod constructors below should be used in preference
            to calling this constructor directly.
         """
        self.desc = desc
    
    def __str__(self):
        return str(self.desc)

    def unify(self, env, other):
        """ Unifies two types.  This may change the type descriptions that they point to.
            If the two types are not unifiable, this will raise an IncompatibleTypes exception
            with the type descriptions that are not compatible.
        """
        if self == other:
            return

        if self.desc == other.desc:
            return

        if isinstance(self.desc, AnonymousVariable):
            self.desc = other.desc
            return
        if isinstance(other.desc, AnonymousVariable):
            other.desc = self.desc
            return
        if isinstance(self.desc, Variable):
            self.desc = other.desc
            return
        if isinstance(other.desc, Variable):
            other.desc = self.desc
            return

        def check_prim1(T, t1, t2):            
            if isinstance(t1, T):
                if not isinstance(t2, T):
                    raise IncompatibleTypes(t1, t2)
                else:
                    return True
            else:
                return False
        
        def check_prims(T, t1, t2):
            return check_prim1(T, t1, t2) or check_prim1(T, t2, t1)


        if check_prims(Int, self.desc, other.desc):
            return
        if check_prims(Real, self.desc, other.desc):
            return

        # At this point, the only types possible should be indexed types
        if not isinstance(self.desc, Indexed) or not isinstance(other.desc, Indexed) :
            raise IncompatibleTypes(self, other)

        if isinstance(self.desc, SomeIndexed):
            self.desc.component().unify(env, other.desc.component())
            self.desc = other.desc
            return
        if isinstance(other.desc, SomeIndexed):
            self.desc.component().unify(env, other.desc.component())
            other.desc = self.desc
            return
        
        # If neither one is a SomeIndexed, then they both need to be 
        # related
        if type(self.desc) is type(other.desc):
            self.desc.component().unify(env,other.desc.component())
        elif issubclass(type(self.desc), type(other.desc)):
            self.desc.component().unify(env, other.desc.component())
            other.desc = self.desc
        elif issubclass(type(other.desc), type(self.desc)):
            self.desc.component().unify(env, other.desc.component())
            self.desc = other.desc
        else:
            raise IncompatibleTypes(self, other)

        

    ### These are the constructor methods for various types
    @classmethod
    def namedVariable(cls, env, name):
        """Create a new type representing a Type Variable (unknown type), in a given environment"""
        if name in env:
            return env[name]
        else:
            v = cls(Variable(name))
            env[name] = v    
            return v

    @classmethod
    def newVariable(cls) -> 'Type_':
        """Create a new type representing a Type Variable (unknown type), in a given environment"""
        return cls(AnonymousVariable())
    
    @classmethod
    def real(cls) -> 'Type_':
        """Create a new type representing a stan Real number"""
        return cls(Real())
    
    @classmethod
    def int(cls) -> 'Type_':
        """Create a new type representing a stan Integer number"""
        return cls(Int())

    @classmethod
    def indexed(cls, component:'Type_'=None) -> 'Type_':
        """Create a new type representing some indexable thing"""
        if not component:
            component = cls.newVariable()
        return cls(SomeIndexed(component))

    @classmethod
    def vector(cls) -> 'Type_':
        """Create a new type representing a stan column vector"""
        return cls(Vector())

    @classmethod
    def row_vector(cls) -> 'Type_':
        """Create a new type representing a stan row vector"""
        return cls(RowVector())

    @classmethod
    def matrix(cls) -> 'Type_':
        """Create a new type representing a stan matrix"""
        return cls(Matrix())

    @classmethod
    def array(cls, component:'Type_') -> 'Type_':
        """Create a new type representing a stan array"""
        return cls(Array(component))

    ### These test methods determine what kind of type it is    
    def isVariable(self) -> bool:
        return isinstance(self.desc, Variable)

    def isPrimitive(self):
        return isinstance(self.desc, Primitive)


class Variable(TypeDesc):
    def __init__(self, name):
        super(Variable, self).__init__()
        self.name = name

    def __str__(self):
        return "?{}".format(self.name)

class AnonymousVariable(Variable):
    @classmethod
    def newVarName(cls):
        if hasattr(cls, 'counter'):
            cls.counter = cls.counter+1
        else:
            cls.counter = 0
        return "anon"+str(cls.counter)
    
    def __init__(self):
        super(AnonymousVariable, self).__init__(AnonymousVariable.newVarName())

    def __str__(self):
        return "?"

class Primitive(TypeDesc):
    def __init__(self):
        super(Primitive, self).__init__()

class Real(Primitive):
    def __init__(self):
        super(Real, self).__init__()

    def __str__(self):
        return "real"

class Int(Primitive):
    def __init__(self):
        super(Int, self).__init__()

    def __str__(self):
        return "int"

class Indexed(TypeDesc):
    """Base class for representing Stan types that can be indexed, such as arrays and vectors"""
    def __init__(self):
        super(Indexed, self).__init__()

#    @abstractmethod
    def component(self) -> Type_:
        pass
    
    def subscript(dims:int):
        c = self
        for _ in range(dims):
            if isinstance(c, Indexed):
                c = c.component()
            else:
                raise ValueError("Attempting to subscript {} with too many dimensions {}".format(self, dims))
        return c

class SomeIndexed(Indexed):
    """Represents some (unknown) stan object that supports indexing"""
    def __init__(self,  component:Type_):
        super(SomeIndexed, self).__init__()
        self.c = component

    def component(self) -> Type_:
        return self.c

    def __str__(self):
        dims = 1
        c = self
        while isinstance(c.component().desc, SomeIndexed):
            c = c.component().desc
            dims = dims + 1
        return "{}[|{}]".format(str(c.component()), ","*(dims-1))

class Vector(Indexed):
    """Represents a Stan column vector"""
    def __init__(self):
        super(Vector, self).__init__()
    
    def component(self) -> Type_:
        return Type_.real()

    def __str__(self):
        return "vector"


class RowVector(Indexed):
    """Represents a Stan row vector"""
    def __init__(self):
        super(RowVector, self).__init__()

    def component(self) -> Type_:
        return Type_.real()
    
    def __str__(self):
        return "row_vector"


class Matrix(Indexed):
    """Represents a Stan matrix"""
    def __init__(self):
        super(Matrix, self).__init__()

    def component(self) -> Type_:
        return Type_(MatrixSlice())

    def __str__(self):
        return "matrix"

class MatrixSlice(Indexed):
    """Represents a slice of a stan matrix (a matrix that has already been subscripted)"""
    def __init__(self):
        super(MatrixSlice, self).__init__()

    def component(self) -> Type_:
        return Type_.real()
    
    def __str__(self):
        return "matrix/slice"

class Array(Indexed):
    """Represents a Stan array"""
    def __init__(self, component:Type_):
        super(Array, self).__init__()
        self.c = component

    def component(self) -> Type_:
        return self.c

    def __str__(self):
        dims = 1
        c = self
        while isinstance(c.component().desc, SomeIndexed):
            c = c.component().desc
            dims = dims + 1
        return "{}[{}]".format(str(c.component()), ","*(dims-1))

