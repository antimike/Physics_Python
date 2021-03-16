class Context:
    class Factory:
        def __init__(self, dependencies, wrapped_constructor):
            self._dependencies = dependencies
            def wrapper(*args, **kwargs):
                kwargs = {**kwargs, **self._dependencies}
                return wrapped_constructor(*args, **kwargs)
            self._constructor_wrapper = wrapper
        def build(self, *args, **kwargs):
            return self._constructor_wrapper(*args, **kwargs)
    def __init__(self, **kwargs):
        self._factories = {}
        self._resources = kwargs
    def load_data(self, data):
        pass
    def register_client(self, client):
        pass
    def register_provider(self, provider):
        pass
    def create_resource_factory(self, client, name):
        reqs = client.needs()
        self._factories[name] = Context.Factory(reqs, client)
        self.safe_add_attr(name, self._factories[name].build)
    def safe_add_attr(self, name, val):
        setattr(self, name, val)
    # Question: Can we treat subcontexts like any other state variable, or are they somehow special?
    def add_subcontext(self):
        pass
    # To be used as a decorator
    def inject(self, func, *dependency_names):
        dependencies = [{name: self._resources[name]} for name in dependency_names]
        def injected(*args, **kwargs):
            return func(*args, **{**dependencies, **kwargs})
        return injected
    # Possibly better name: "do"
    def add_decorator(self, context_decorator):
        pass

def get_nodes(obj):
    ret = {}
    try:
        for key, val in obj.items():
            ret[key] = {
                'FORWARD': set(val.keys()),
                'BACK': set()
            }
            {for node in get_nodes(val)}

class Node:
    def __init__(self, name):
        self._name = name
        self._parents = set()
        self._children = set()
    @property
    def name(self):
        return self._name
    @property
    def parents(self):
        return self._parents
    @property
    def children(self):
        return self._children
    # TODO: Consistency checks?
    def add_parents(self, parents):
        self._parents |= parents
    def add_children(self, children):
        self._children |= children

def get_dict(x, parent):
    if type(x) is dict:
        return x
    else:
        return {parent: x}

def merge(x, y, parent):
    # Double 'None' check
    # Might be better to be explicit
    if not x and y:
        return x or y
    xd, yd = get_dict(x, parent), get_dict(y, parent)
    # Effectively asserts that the last argument shouldn't matter
    return deep_merge(xd, yd, None)

def deep_merge(x, y, parent):
    # Base case
    if not type(x) is dict and not type(y) is dict:
        try:
            return x | y
        except:
            return {x, y}
    # Recursive case
    elif type(x) is dict and type(y) is dict:
        return {
            k: merge(x.get(k), y.get(k), k)
            for k in x.keys() | y.keys()
        }
    # Covers case one or the other is not a dict
    # Helper function just constructs "leaf dict" to insert into the other
    else:
        return merge(x, y, parent)

class Contextual:
    def __init__(self, context, **dependency_dict):
        self._context = context
        self._needs = dependency_dict.get("needs", [])
        self._provides = dependency_dict.get("provides", [])
        if context is not None:
            context.register(self)
        # Is this necessary?  Or advisable?
        for dep in self._needs:
            setattr(self, dep, lambda : self.context.get(dep))
    @property
    def needs(self):
        return self._needs
    @property
    def provides(self):
        return self._provides
    @property
    def context(self):
        return self._context

