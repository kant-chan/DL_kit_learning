class ObjectDict(dict):
    def __init__(self, *args, **kwargs):
        super(ObjectDict, self).__init__(*args, **kwargs)

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = ObjectDict(value)
        return value

# if __name__ == '__main__':
#     a = {'name': 'muzhi'}
#     # print(getattr(a, 'name'))    # AttributeError: 'dict' object has no attribute 'name'
#     od = ObjectDict(asf={'a': 1}, d=True)
#     print(od.asf, od.asf.a)
#     print(od.d)
#     print(getattr(od.asf, 'a'))


#############################
class WidgetShowLazyLoad(object):
    def fetch_complex_attr(self, attrname):
        '''可能是比较耗时的操作， 比如从文件读取'''
        return attrname

    def __getattr__(self, name):
        if name not in self.__dict__:
             self.__dict__[name] = self.fetch_complex_attr(name)
        return self.__dict__[name]

# if __name__ == '__main__':
#     w = WidgetShowLazyLoad()
#     print 'before', w.__dict__
#     w.lazy_loaded_attr
#     print 'after', w.__dict__


#############################
class adaptee(object):
    def foo(self):
        print('foo in adaptee')
    def bar(self):
        print('bar in adaptee')

class adapter(object):
    def __init__(self):
        self.adaptee = adaptee()

    def foo(self):
        print('foo in adapter')
        self.adaptee.foo()

    def __getattr__(self, name):
        return getattr(self.adaptee, name)

# if __name__ == '__main__':
#     a = adapter()
#     a.foo()
#     a.bar()


#############################
class AlgoImpA(object):
    def __init__(self):
        self.obj_attr = 'obj_attr in AlgoImpA'

    def foo(self):
        print('foo in AlgoImpA')

    def bar(self):
        print('bar in AlgoImpA')

class AlgoImpB(object):
    def __init__(self):
        self.obj_attr = 'obj_attr in AlgoImpB'

    def foo(self):
        print('foo in AlgoImpB')

    def bar(self):
        print('bar in AlgoImpB')

class Algo(object):
    def __init__(self):
        self.imp_a = AlgoImpA()
        self.imp_b = AlgoImpB()
        self.cur_imp = self.imp_a

    def switch_imp(self):
        if self.cur_imp == self.imp_a:
            self.cur_imp = self.imp_b
        else:
            self.cur_imp = self.imp_a

    def __str__(self):
        return 'Algo with imp %s' % str(self.cur_imp)


    def __getattr__(self, name):
        return getattr(self.cur_imp, name)


# if __name__ == '__main__':
#     algo = Algo()
    
#     print(algo)
#     print algo.obj_attr
#     algo.foo()
    
#     algo.switch_imp()
    
#     print algo
#     print algo.obj_attr
#     algo.bar()

"""
magic method
"""

class Test:
    def __getattr__(self, name):
        print('__getattr__')

    def __getattribute__(self, name):
        print('__getattribute__')

    def __setattr__(self, name, value):
        print('__setattr__')

    def __delattr__(self, name):
        print('__delattr__')

t = Test()
t.x = 0

#####################
class Parent(object):
    a = 0
    b = 1
 
    def __init__(self):
        self.a = 2
        self.b = 3
 
    def p_test(self):
        pass
 
 
class Child(Parent):
    a = 4
    b = 5
 
    def __init__(self):
        super(Child, self).__init__()
        # self.a = 6
        # self.b = 7
 
    def c_test(self):
        pass
 
    def p_test(self):
        pass

p = Parent()
c = Child()
print(Parent.__dict__)
print(Child.__dict__)
print(p.__dict__)
print(c.__dict__)
