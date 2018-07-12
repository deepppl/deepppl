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
    pass


class MissingPriorNetException(TranslationException):
    def __init__(self, net, params):
        str = "The following parameters of {} were note given a prior:{}"
        msg = str.format(net, params)
        super(MissingPriorNetException, self).__init__(msg)
        self.net = net
        self.params = params

class MissingGuideNetException(TranslationException):
    def __init__(self, net, params):
        str = "The following parameters of {} were note given a guide:{}"
        msg = str.format(net, params)
        super(MissingGuideNetException, self).__init__(msg)
        self.net = net
        self.params = params

class MissingModelExeption(TranslationException):
    def __init__(self, latents):
        str = "The following latents {} were not sampled on the model."
        msg = str.format(latents)
        super(MissingModelExeption, self).__init__(msg)
        self.latents = latents

class MissingGuideExeption(TranslationException):
    def __init__(self, latents):
        str = "The following latents {} were not sampled on the guide."
        msg = str.format(latents)
        super(MissingGuideExeption, self).__init__(msg)
        self.latents = latents

class ObserveOnGuideExeption(TranslationException):
    def __init__(self, data):
        str = "Trying to observer data {} inside the guide."
        msg = str.format(data)
        super(ObserveOnGuideExeption, self).__init__(msg)
        self.data = data
