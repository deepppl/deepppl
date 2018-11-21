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
        self._desc = desc
    
    def __str__(self):
        return str(self.description())

    def __repr__(self):
        return f"D{repr(self.description())}"

    def unify(self, other:'Dimension', *, equalities:Set[Tuple['KnownDimension', 'KnownDimension']]=set()):
        """ Unifies two dimensions.  This may change the dimension descriptions that they point to.
            If the two dimensions are not unifiable, this will raise an IncompatibleDimensions exception
            with the dimensions descriptions that are not compatible.
        """
        if self == other:
            return

        me = self.target()
        other = other.target()

        if me == other:
            return

        if me._desc == other._desc:
            return

        if isinstance(me._desc, AnonymousDimensionVariable):
            me._desc = DimensionLink(other)
            return
        if isinstance(other._desc, AnonymousDimensionVariable):
            other._desc = DimensionLink(me)
            return
        if isinstance(me._desc, DimensionVariable):
            me._desc = DimensionLink(other)
            return
        if isinstance(other._desc, DimensionVariable):
            other._desc = DimensionLink(me)
            return

        assert isinstance(me._desc, KnownDimension)
        assert isinstance(other._desc, KnownDimension)

        if type(me._desc) == type(other._desc) and me._desc.expr() == other._desc.expr():
            return
        if isinstance(me._desc, ConstantDimension) and isinstance(other._desc, ConstantDimension):
            c1 = ast.literal_eval(me._desc.expr())
            c2 = ast.literal_eval(other._desc.expr())
            if c1 != c2:
                raise IncompatibleDimensions(self, other)
        if not (other._desc,me._desc) in equalities:
            equalities.add((me._desc,other._desc))

        # # If one of them is a constant, we will enforce that the other is as well.
        # # We could also imagine allowing it unify with a pathDimension, 
        # # along with a runtime check for equality
        # if isinstance(me._desc, ConstantDimension):
        #     if isinstance(other._desc, ConstantDimension):
        #         if me._desc.val == other._desc.val:
        #             return
        #         else:
        #             raise IncompatibleDimensions(self, other)
        #     else:
        #         raise IncompatibleDimensions(self, other)
        # if isinstance(other._desc, ConstantDimension):
        #     if isinstance(me._desc, ConstantDimension):
        #         if me._desc.val == other._desc.val:
        #             return
        #         else:
        #             raise IncompatibleDimensions(self, other)
        #     else:
        #         raise IncompatibleDimensions(self, other)

        # # At this point, the only dimensions possible should be path dimension types
        # if not isinstance(me._desc, PathDimension) or not isinstance(other._desc, PathDimension):
        #     raise IncompatibleDimensions(self, other)

        # if me._desc.path == other._desc.path:
        #     # if they are the same path, we may as well identify them
        #     me._desc = other._desc
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
    def shapeDimension(cls, path, index=None) -> 'Dimension':
        """Create a new dimension representing the shape of another runtime expression (e.g. encoder[1]$shape)"""
        if index is not None:
            return cls(SubscriptedShapeDimension(path, index))
        else:
            return cls(ShapeDimension(path))


    @classmethod
    def runtimeDimension(cls, e) -> 'Dimension':
        """Create a new dimension representing a value known at runtime (e.g. x)"""
        return cls(RuntimeDimension(e))


    @classmethod
    def constantDimension(cls, val) -> 'Dimension':
        """Create a new dimension representing a constant (e.g. 2)"""
        return cls(ConstantDimension(val))

    ### This method is used to find the "actual" dimension (following links as needed)
    def target(self) -> 'Dimension':
        """return the concrete (non-link) dimension, following links as needed"""
        if isinstance(self._desc, DimensionLink):
            return self._desc.target()
        else:
            return self

    def description(self) -> DimensionDesc:
        """return the description for this dimension, following links as needed"""
        return self.target()._desc

    ### These methods check what type of dimension this is
    def isVariable(self):
        """Does this dimension point to a (possibly anonymous) dimension variable?"""
        return isinstance(self.description(), DimensionVariable)

    def isConstant(self):
        """Does this dimension point to a constant dimension?"""
        return isinstance(self.description(), ConstantDimension)

    def isKnown(self):
        """Does this dimension point to a known dimension?"""
        return isinstance(self.description(), KnownDimension)


class DimensionLink(DimensionDesc):
    def __init__(self, target:Dimension):
        super(DimensionLink, self).__init__()
        self._target = target
    
    def target(self):
        # path compression
        self._target = self._target.target()
        return self._target


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

class SubscriptedShapeDimension(ShapeDimension):
    """Represents a dimension that should be the same as a component of the shape of a runtime expression"""
    def __init__(self, path, index):
        super(SubscriptedShapeDimension, self).__init__(path)
        self.index = index
    
    def __str__(self):
        return f"{self.path}$shape[{self.index}]"

    def expr(self)->str:
        return f"{self.path}.shape[{self.index}]"

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