from langchain.tools import tool



@tool
def get_greetings(name : str ) -> str:
   """Generate a greeting message for a user"""
   
   return f"Hello {name}, Welcome to the AI worls"

result = get_greetings.invoke({"name" : "Neha"})
print (result)

print(get_greetings.name)
print(get_greetings.description)
print(get_greetings.args)
