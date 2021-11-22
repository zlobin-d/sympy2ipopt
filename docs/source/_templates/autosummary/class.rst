{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% set methods = methods|reject("in", inherited_members)|reject("eq", "__init__")|sort %}
   {% if "__len__" in members %}
      {{ methods.insert(0, "__len__") or '' }}
   {% endif %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set attributes = attributes|reject("in", inherited_members)|reject("eq", "default_assumptions")|sort %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
