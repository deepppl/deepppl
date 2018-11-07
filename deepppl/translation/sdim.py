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

    def unify(self, other:'Dimension', *, denv={}):
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

        # If one of them is a constant, we will enforce that the other is as well.
        # We could also imagine allowing it unify with a pathDimension, 
        # along with a runtime check for equality
        if isinstance(self.desc, ConstantDimension):
            if isinstance(other.desc, ConstantDimension):
                if self.desc.val == other.desc.val:
                    return
                else:
                    raise IncompatibleDimensions(self, other)
            else:
                raise IncompatibleDimensions(self, other)
        if isinstance(other.desc, ConstantDimension):
            if isinstance(self.desc, ConstantDimension):
                if self.desc.val == other.desc.val:
                    return
                else:
                    raise IncompatibleDimensions(self, other)
            else:
                raise IncompatibleDimensions(self, other)

        # At this point, the only dimensions possible should be path dimension types
        if not isinstance(self.desc, PathDimension) or not isinstance(other.desc, PathDimension):
            raise IncompatibleDimensions(self, other)

        if self.desc.path == other.desc.path:
            # if they are the same path, we may as well identify them
            self.desc = other.desc
            return
        else:
            # What do we do if we have two paths for the same shape?  For now,
            # We flag this as an error.  A (probably better) alternative would be to allow it,
            # but check that the shape constraint holds at runtime.
            raise IncompatibleDimensions(self, other)

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
    def pathDimension(cls, path) -> 'Dimension':
        """Create a new dimension representing a path from a known runtime entity (e.g. an MLP)"""
        return cls(PathDimension(path))

    @classmethod
    def constantDimension(cls, val) -> 'Dimension':
        """Create a new dimension representing a constant (e.g. 2)"""
        return cls(ConstantDimension(val))

class DimensionVariable(DimensionDesc):
    def __init__(self, name):
        super(DimensionVariable, self).__init__()
        self.name = name

    def __str__(self):
        return "?{}".format(self.name)

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

class PathDimension(DimensionDesc):
    def __init__(self, path):
        super(PathDimension, self).__init__()
        self.path = path
    
    def __str__(self):
        return "${}".format(self.path)

class ConstantDimension(DimensionDesc):
    def __init__(self, val):
        super(ConstantDimension, self).__init__()
        self.val = val
    
    def __str__(self):
        return str(self.val)

Dnew = Dimension.newVariable
Dnamed = Dimension.namedVariable
Dpath = Dimension.pathDimension
Dconstant = Dimension.constantDimension