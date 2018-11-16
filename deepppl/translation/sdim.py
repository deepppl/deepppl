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

from typing import Set, Tuple
import ast

from .exceptions import IncompatibleDimensions

# This defines a dimension system for Stan, extended with support for Dimension Variables
# This will be used by the Type System

class DimensionDesc(object):
    """ This is the (abstract) base class for dimension descriptions.
        These are wrapped by Dimension objects, which are generally what you want to use.
    """
    pass

class Dimension(object):
    """ Creates a new dimension object, which represents the dimension of something that is Indexed.
        It contains a dimension description, describing the known information about the described dimension.
        This dimension description may be shared with other Dimension objects, indicating 
        that the two dimensions are the same.
    """
    def __init__(self, desc:DimensionDesc):
        """ Creates a new Dimension, wrapping a given dimension description (representation).
            Generally, the classmethod constructors below should be used in preference
            to calling this constructor directly.
         """
        self.desc = desc
    
    def __str__(self):
        return str(self.desc)

    def __repr__(self):
        return f"D{repr(self.desc)}"

    def unify(self, other:'Dimension', *, equalities:Set[Tuple['KnownDimension', 'KnownDimension']]=set()):
        """ Unifies two dimensions.  This may change the dimension descriptions that they point to.
            If the two dimensions are not unifiable, this will raise an IncompatibleDimensions exception
            with the dimensions descriptions that are not compatible.
        """
        if self == other:
            return

        if self.desc == other.desc:
            return

        if isinstance(self.desc, AnonymousDimensionVariable):
            self.desc = other.desc
            return
        if isinstance(other.desc, AnonymousDimensionVariable):
            other.desc = self.desc
            return
        if isinstance(self.desc, DimensionVariable):
            self.desc = other.desc
            return
        if isinstance(other.desc, DimensionVariable):
            other.desc = self.desc
            return

        assert isinstance(self.desc, KnownDimension)
        assert isinstance(other.desc, KnownDimension)

        if type(self.desc) == type(other.desc) and self.desc.expr() == other.desc.expr():
            return
        if isinstance(self.desc, ConstantDimension) and isinstance(other.desc, ConstantDimension):
            c1 = ast.literal_eval(self.desc.expr())
            c2 = ast.literal_eval(other.desc.expr())
            if c1 != c2:
                raise IncompatibleDimensions(self, other)
        if not (other.desc,self.desc) in equalities:
            equalities.add((self.desc,other.desc))

        # # If one of them is a constant, we will enforce that the other is as well.
        # # We could also imagine allowing it unify with a pathDimension, 
        # # along with a runtime check for equality
        # if isinstance(self.desc, ConstantDimension):
        #     if isinstance(other.desc, ConstantDimension):
        #         if self.desc.val == other.desc.val:
        #             return
        #         else:
        #             raise IncompatibleDimensions(self, other)
        #     else:
        #         raise IncompatibleDimensions(self, other)
        # if isinstance(other.desc, ConstantDimension):
        #     if isinstance(self.desc, ConstantDimension):
        #         if self.desc.val == other.desc.val:
        #             return
        #         else:
        #             raise IncompatibleDimensions(self, other)
        #     else:
        #         raise IncompatibleDimensions(self, other)

        # # At this point, the only dimensions possible should be path dimension types
        # if not isinstance(self.desc, PathDimension) or not isinstance(other.desc, PathDimension):
        #     raise IncompatibleDimensions(self, other)

        # if self.desc.path == other.desc.path:
        #     # if they are the same path, we may as well identify them
        #     self.desc = other.desc
        #     return
        # else:
        #     # What do we do if we have two paths for the same shape?  For now,
        #     # We flag this as an error.  A (probably better) alternative would be to allow it,
        #     # but check that the shape constraint holds at runtime.
        #     raise IncompatibleDimensions(self, other)

    ### These are the constructor methods for various types
    @classmethod
    def namedVariable(cls, denv, name):
        """Create a new dimension representing a Dimension Variable (unknown dimension), in a given environment"""
        if name in denv:
            return denv[name]
        else:
            v = cls(DimensionVariable(name))
            denv[name] = v    
            return v

    @classmethod
    def newVariable(cls) -> 'Dimension':
        """Create a new dimension representing a Dimension Variable (unknown dimension), in a given environment"""
        return cls(AnonymousDimensionVariable())
    
    @classmethod
    def shapeDimension(cls, path) -> 'Dimension':
        """Create a new dimension representing the shape of another runtime expression (e.g. encoder[1]$shape)"""
        return cls(ShapeDimension(path))

    @classmethod
    def runtimeDimension(cls, e) -> 'Dimension':
        """Create a new dimension representing a value known at runtime (e.g. x)"""
        return cls(RuntimeDimension(e))


    @classmethod
    def constantDimension(cls, val) -> 'Dimension':
        """Create a new dimension representing a constant (e.g. 2)"""
        return cls(ConstantDimension(val))

    ### These methods check what type of dimension this is
    def isVariable(self):
        """Does this dimension point to a (possibly anonymous) dimension variable?"""
        return isinstance(self.desc, DimensionVariable)

    def isConstant(self):
        """Does this dimension point to a constant dimension?"""
        return isinstance(self.desc, ConstantDimension)

    def isKnown(self):
        """Does this dimension point to a known dimension?"""
        return isinstance(self.desc, KnownDimension)


class DimensionVariable(DimensionDesc):
    def __init__(self, name):
        super(DimensionVariable, self).__init__()
        self.name = name

    def __str__(self):
        return "?{}".format(self.name)

    __repr__ = __str__

class AnonymousDimensionVariable(DimensionVariable):
    @classmethod
    def newVarName(cls):
        if hasattr(cls, 'counter'):
            cls.counter = cls.counter+1
        else:
            cls.counter = 0
        return "shape"+str(cls.counter)
    
    def __init__(self):
        super(AnonymousDimensionVariable, self).__init__(AnonymousDimensionVariable.newVarName())

    def __str__(self):
        return "?"

    __repr__ = __str__

class KnownDimension(DimensionDesc):
    """ Represents a dimension whose value is `known`, meaning
        that we have an expression for it that can be determined at runtime
    """
    def __init__(self):
        super(KnownDimension, self).__init__()
    
    def expr(self):
        """An expression that (at runtime) evaluates to the required dimension (integer)"""
        pass

class ShapeDimension(KnownDimension):
    """Represents a dimension that should be the same as the shape of runtime expression"""
    def __init__(self, path):
        super(ShapeDimension, self).__init__()
        self.path = path
    
    def __str__(self):
        return f"{self.path}$shape"

    def expr(self)->str:
        return f"{self.path}.shape"

    __repr__ = __str__

class RuntimeDimension(KnownDimension):
    """A dimension that is the value of evaluating a runtime expression """
    def __init__(self, e):
        super(RuntimeDimension, self).__init__()
        self.e = e
    
    def __str__(self):
        return str(self.e)

    def expr(self)->str:
        return str(self.e)

    __repr__ = __str__

class ConstantDimension(RuntimeDimension):
    """A dimension that is known statically """
    def __init__(self, val):
        super(ConstantDimension, self).__init__(val)
    
    def __str__(self):
        return str(self.e)

    __repr__ = __str__

Dnew = Dimension.newVariable
Dnamed = Dimension.namedVariable
Dshape = Dimension.shapeDimension
Druntime = Dimension.runtimeDimension
Dconstant = Dimension.constantDimension