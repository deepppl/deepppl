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

from typing import Union, List, Set, Tuple
from .exceptions import IncompatibleTypes
from .sdim import Dimension, KnownDimension

# This defines a type system for Stan, extended with support for Type Variables
# We do not currently model constraints.

class TypeDesc(object):
    """ This is the (abstract) base class for type descriptions.
        These are wrapped by Type_ objects, which are generally what you want to use.
    """
    pass

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

    def __repr__(self):
        return f"T{repr(self.desc)}"

    def unify(self, other, *, equalities:Set[Tuple[KnownDimension, KnownDimension]]=set(), tenv={}):
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

        ## TODO: fix this to handle Primitive types (which can be either Integer or Reals)
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

        self.desc.dimension.unify(other.desc.dimension, equalities=equalities)

        if isinstance(self.desc, SomeIndexed):
            self.desc.component().unify(other.desc.component(), equalities=equalities, tenv=tenv)
            self.desc = other.desc
            return
        if isinstance(other.desc, SomeIndexed):
            self.desc.component().unify(other.desc.component(), equalities=equalities, tenv=tenv)
            other.desc = self.desc
            return
        
        # If neither one is a SomeIndexed, then they both need to be 
        # related
        if type(self.desc) is type(other.desc):
            self.desc.component().unify(other.desc.component(), equalities=equalities, tenv=tenv)
        elif issubclass(type(self.desc), type(other.desc)):
            self.desc.component().unify(other.desc.component(), equalities=equalities, tenv=tenv)
            other.desc = self.desc
        elif issubclass(type(other.desc), type(self.desc)):
            self.desc.component().unify(other.desc.component(), equalities=equalities, tenv=tenv)
            self.desc = other.desc
        else:
            raise IncompatibleTypes(self, other)

        

    ### These are the constructor methods for various types
    @classmethod
    def namedVariable(cls, tenv, name):
        """Create a new type representing a Type Variable (unknown type), in a given environment"""
        if name in tenv:
            return tenv[name]
        else:
            v = cls(Variable(name))
            tenv[name] = v    
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
    def indexed(cls, component:'Type_'=None, *, dimension:Dimension=None) -> 'Type_':
        """Create a new type representing some indexable thing"""
        if not component:
            component = cls.newVariable()
        if not dimension:
            dimension = Dimension.newVariable()
        return cls(SomeIndexed(dimension, component))

    @classmethod
    def vector(cls, *, dimension:Dimension=None) -> 'Type_':
        """Create a new type representing a stan column vector"""
        if not dimension:
            dimension = Dimension.newVariable()

        return cls(Vector(dimension))

    @classmethod
    def row_vector(cls, *, dimension:Dimension=None) -> 'Type_':
        """Create a new type representing a stan row vector"""
        if not dimension:
            dimension = Dimension.newVariable()

        return cls(RowVector(dimension))

    @classmethod
    def matrix(cls, *, dimension1:Dimension=None, dimension2:Dimension=None) -> 'Type_':
        """Create a new type representing a stan matrix"""
        if not dimension1:
            dimension1 = Dimension.newVariable()

        if not dimension2:
            dimension2 = Dimension.newVariable()

        return cls(Matrix(dimension1, dimension2))

    @classmethod
    def array(cls, component:'Type_'=None, *, dimension:Union[Dimension, List[Dimension]]=None) -> 'Type_':
        """Create a new type representing a stan array"""
        if not component:
            component = cls.newVariable()

        if not dimension:
            dimension = Dimension.newVariable()

        if isinstance(dimension, list):
            c = component
            for dim in dimension:
                c = cls.array(dimension=dim, component=c)
            return c
        else:
            return cls(Array(dimension, component))

    ### These test methods determine what kind of type it is    
    def isVariable(self) -> bool:
        return isinstance(self.desc, Variable)

    def isPrimitive(self):
        return isinstance(self.desc, Primitive)

    def isInt(self):
        return isinstance(self.desc, Int)

    def isReal(self):
        return isinstance(self.desc, Real)

    def isArray(self):
        return isinstance(self.desc, Array)

    def dimensions(self) -> List[Dimension] :
        ## TODO: Note again the choice that only arrays have "dimensions".  This may need to be revisited.
        if isinstance(self.desc, Array):
            return self.desc.dimensions()
        else:
            return list()

    ### These methods take a type and validate that it contains a certain shape.
    ### They may also return a type with a related shape
    def asRealArray(self):
        """ Checks that this is an array type with a base that is either a real or an int,
            and returns an array type with the same dimensions but over a real"""
        # Known limitation: for anonymous types, we don't work so well :-)
        if self.isVariable():
            return self
        if self.isReal():
            return self
        elif self.isInt():
            return self.real()
        elif self.isArray():
            t = self.desc.component().asRealArray()
            if t is self.desc.component():
                return self
            else:
                return self.array(component=t, dimension=self.desc.dimension)
        else:
            raise IncompatibleTypes(self, [Type_.real(), Type_.int()])

    def realArrayToIntArray(self):
        """ Checks that this is an array type with a base that is either a real or an int,
            and returns an array type with the same dimensions but over a real"""
        # Known limitation: for anonymous types, we don't work so well :-)
        if self.isVariable():
            return self
        if self.isReal():
            return self.int()
        elif self.isInt():
            # does bernoulli accept an integer argument?
            raise IncompatibleTypes(self, [Type_.real()])
        elif self.isArray():
            t = self.desc.component().realArrayToIntArray()
            if t is self.desc.component():
                return self
            else:
                return self.array(component=t, dimension=self.desc.dimension)
        else:
            raise IncompatibleTypes(self, [Type_.real(), Type_.int()])




class Variable(TypeDesc):
    def __init__(self, name):
        super(Variable, self).__init__()
        self.name = name

    def __str__(self):
        return "?{}".format(self.name)

    def __repr__(self):
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

    __repr__ = __str__


class Primitive(TypeDesc):
    def __init__(self):
        super(Primitive, self).__init__()

class Real(Primitive):
    def __init__(self):
        super(Real, self).__init__()

    def __str__(self):
        return "real"

    __repr__ = __str__

class Int(Primitive):
    def __init__(self):
        super(Int, self).__init__()

    def __str__(self):
        return "int"

    __repr__ = __str__


class Indexed(TypeDesc):
    """Base class for representing Stan types that can be indexed, such as arrays and vectors"""
    def __init__(self, dimension:Dimension):
        super(Indexed, self).__init__()
        self.dimension = dimension

#    @abstractmethod
    def component(self) -> Type_:
        pass
    
    """ def subscript(dims:int):
        c = self
        for _ in range(dims):
            if isinstance(c, Indexed):
                c = c.component()
            else:
                raise ValueError("Attempting to subscript {} with too many dimensions {}".format(self, dims))
        return c
 """
class SomeIndexed(Indexed):
    """Represents some (unknown) stan object that supports indexing"""
    def __init__(self,  dimension:Dimension, component:Type_):
        super(SomeIndexed, self).__init__(dimension)
        self.c = component

    def component(self) -> Type_:
        return self.c

    def __str__(self):
        s = "[!"
        c = self
        while isinstance(c.component().desc, Array):
            s = s + str(c.dimension) + ","
            c = c.component().desc
        s = str(c.component()) + s + str(c.dimension) + "]"
        return s

    __repr__ = __str__

class Vector(Indexed):
    """Represents a Stan column vector"""
    def __init__(self, dimension:Dimension):
        super(Vector, self).__init__(dimension)
    
    def component(self) -> Type_:
        return Type_.real()

    def __str__(self):
        return "vector"

    __repr__ = __str__

class RowVector(Indexed):
    """Represents a Stan row vector"""
    def __init__(self, dimension:Dimension):
        super(RowVector, self).__init__(dimension)

    def component(self) -> Type_:
        return Type_.real()
    
    def __str__(self):
        return "row_vector"

    __repr__ = __str__

class MatrixSlice(Indexed):
    """Represents a slice of a stan matrix (a matrix that has already been subscripted)"""
    def __init__(self, dimension:Dimension):
        super(MatrixSlice, self).__init__(dimension)

    def component(self) -> Type_:
        return Type_.real()
    
    def __str__(self):
        return "matrix/slice[{}]".format(self.dimension)

    __repr__ = __str__

class Matrix(Indexed):
    """Represents a Stan matrix"""
    def __init__(self, dimension1:Dimension, dimension2:Dimension):
        super(Matrix, self).__init__(dimension1)
        self.sliceDim = dimension2

    def component(self) -> Type_:
        return Type_(MatrixSlice(self.sliceDim))

    def __str__(self):
        return "matrix[{},{}]".format(self.dimension, self.sliceDim)

    __repr__ = __str__

class Array(Indexed):
    """Represents a Stan array"""
    def __init__(self, dimension:Dimension, component:Type_):
        super(Array, self).__init__(dimension)
        self.c = component

    def component(self) -> Type_:
        return self.c

    def dimensions(self) -> List[Dimension]:
        res = list()
        res.append(self.dimension)
        c = self
        while isinstance(c.component().desc, Array):
            c = c.component().desc
            res.append(c.dimension)
        return res

    def __str__(self):
        s =  "["
        c = self
        while isinstance(c.component().desc, Array):
            s = s + str(c.dimension) + ","
            c = c.component().desc
        s = str(c.component()) + s + str(c.dimension) + "]"
        return s

    __repr__ = __str__

Tnamed = Type_.namedVariable
Tnew = Type_.newVariable
Treal = Type_.real
Tint = Type_.int
Tindexed = Type_.indexed
Tvector = Type_.vector
Trow_vector = Type_.row_vector
Tmatrix = Type_.matrix
Tarray = Type_.array