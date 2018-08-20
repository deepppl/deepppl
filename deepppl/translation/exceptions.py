# /*
#  * Copyright 2018 IBM Corporation
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  * http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
# */

class DeepPPLException(Exception):
    pass


class TranslationException(DeepPPLException):
    def __init__(self, *args):
        msg = self._base_msg.format(*args)
        super(TranslationException, self).__init__(msg)
        self.args = args
    


class MissingPriorNetException(TranslationException):
    _base_msg = "The following parameters of {} were note given a prior:{}"

class MissingGuideNetException(TranslationException):
    _base_msg = "The following parameters of {} were note given a guide:{}"

class MissingModelExeption(TranslationException):
    _base_msg = "The following latents {} were not sampled on the model."

class MissingGuideExeption(TranslationException):
    _base_msg = "The following latents {} were not sampled on the guide."

class ObserveOnGuideExeption(TranslationException):
    _base_msg = "Trying to observer data {} inside the guide."
    
class UnsupportedProperty(TranslationException):
    _base_msg = "Unsupported property: {}."

class UndeclaredParametersException(TranslationException):
    _base_msg = "Use of undeclared parameters: {}."

class UndeclaredNetworkException(TranslationException):
    _base_msg = "Use of undeclared network: {}."

class InvalidSamplingException(TranslationException):
    _base_msg = "Only identifiers and indexing are supported as lhs of sampling: {}."

class UndeclaredVariableException(TranslationException):
    _base_msg = "Undeclared identifier: {}."

class UnknownDistributionException(TranslationException):
    _base_msg = "Unknown distribution: {}."

class AlreadyDeclaredException(TranslationException):
    _base_msg = "Variable '{}' already declared."

class IncompatibleShapes(TranslationException):
    _base_msg = "Trying to use incompatible shapes:{} and {}"
    