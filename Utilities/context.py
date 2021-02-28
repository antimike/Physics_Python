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

