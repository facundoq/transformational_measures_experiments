import abc
from typing import Generic,TypeVar,Any
from typing import *
import typing



class Parser(abc.ABC):
    @abc.abstractmethod
    def parse(self,s:str)->(str,Any):
        pass
    def parse_end(self,s:str)->Any:
        s,v=self.parse(s)
        if len(s)>0:
            raise ParserFailed(f"Invalid extra string {s}.")
        return v
    def parse_const(self,s:str,const:str)->str:
        if s.startswith(const):
            return s[len(const):]
        raise ParserFailed(f"Failed to find {const} at the start of {s}.")

    def parse_const_end(self,s:str,const:str):
        if s.endswith(const):
            return s[:-len(const)]
        raise ParserFailed(f"Failed to find {const} at the end of {s}.")

class ParserFailed(Exception):
    ''' Signals a parsing failure, with a corresponding message to show the user.'''

def eval_all(s:str,parsers:[Parser])->(str,Any):
    for p in parsers:
        try:
            a=p.parse(s)
            return a
        except ParserFailed:
            pass
        raise ParserFailed

class KeyValueParameterParser(Parser):
    def __init__(self,name:str,optional=False,delimiters:[str]=None):
        self.name=name
        self.optional=optional
        if delimiters is None:
            delimiters =[",", ")"]
        self.delimiters = delimiters

    def parse(self,s:str)->(str,Any):
        s = self.parse_const(s, self.name)
        s = self.parse_const(s,"=")
        s,v = self.parse_value(s)
        return s,v

    @abc.abstractmethod
    def parse_value(self,s:str):
        pass

class SimpleValueParser(KeyValueParameterParser):
    @abc.abstractmethod
    def parse_simple_value(self, s:str)->Any:
        pass

    def parse_value(self,s:str)->(str,int):
        i = 0
        while i < len(s) and (not s[i] in self.delimiters):
            i += 1
        if i == 0:
            raise ParserFailed(f"Empty value")
        s_value, s = s[:i], s[i:]
        return s,self.parse_simple_value(s_value)

class StringParser(SimpleValueParser):
    def parse_value(self,s:str)->(str,str):
        s = self.parse_const(s,'"')
        i=0
        while i < len(s) and s != '"':
            i += 1
        s_value, s = s[:i], s[i:]
        s = self.parse_const(s,'"')
        return s,s_value

class IntParser(SimpleValueParser):
    def parse_simple_value(self,s:str)->int:
        try:
            return int(s)
        except ValueError:
            raise ParserFailed(f'Could not parse "{s}" as an integer number.')


class FloatParser(SimpleValueParser):
    def parse_simple_value(self,s:str)->float:
        try:
            return  float(s)
        except ValueError:
            raise ParserFailed(f'Could not parse "{s}" as an floating point number.')




import transformation_measure as tm

class ClassParser(Parser):
    def __init__(self, klass:type, parameters:[KeyValueParameterParser], short_name:str=None):
        self.klass:type = klass
        self.name:str = klass.__name__
        self.short_name:str = short_name
        self.parameter_parsers:[KeyValueParameterParser]=parameters

    def parse_name(self, s: str) -> str:
        if s.startswith(self.name) :
            return s[:len(self.name)]
        elif not self.short_name is None and s.startswith(self.short_name):
            return s[:len(self.short_name)]
        else:
            if self.short_name is None:
                short_message=""
            else:
                short_message=f" or {self.short_name}"
            raise ParserFailed(f"String {s} did not start with {self.name}{short_message}.")

    def parse_parameters(self,s:str)->(str,Dict[str,Any]):
        parameters={}
        for p in self.parameter_parsers:
            try:
                s,o=p.parse(s)
                s = self.parse_const(s,",")
                parameters[p.name] = o
            except ParserFailed as pf:
                if not p.optional:
                    raise ParserFailed(f"Could not parse parameter {p.name}, reason: {pf} ")

        return s,parameters

    def parse(self,s:str)->(str,Any):
        s = self.parse_name(s)
        s = self.parse_const(s, "(")
        s,parameters = self.parse_parameters(s)
        s = self.parse_const(s, ")")
        return s,self.create_object(parameters)

    @abc.abstractmethod
    def create_object(self,parameters:Dict[str,Any]):
        pass


class AffineTransformationParser(ClassParser):
    def __init__(self):
        parameters=[
                    IntParser("r"),
                    IntParser("s"),
                    IntParser("t"),
                    ]
        super().__init__(self.__class__,parameters,short_name="Affine")

    def parse(self,s:str)->(str,tm.TransformationSet):
        return super().parse(s)


    def create_object(self,p:Dict[str,Any]):
        r,s,t=p["r"],p["s"],p["t"]
        return tm.SimpleAffineTransformationGenerator(r=r,s=s,t=t)

