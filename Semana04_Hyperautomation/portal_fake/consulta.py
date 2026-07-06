# consulta.py

usuarios = [
    {"nome": "Maria Silva", "email": "maria@email.com"},
    {"nome": "João Santos", "email": "joao@email.com"}
]

def consultar_usuarios():
    print("=== Lista de Usuários ===")

    if not usuarios:
        print("Nenhum usuário cadastrado.")
        return

    for usuario in usuarios:
        print(f"Nome: {usuario['nome']} | E-mail: {usuario['email']}")

# Teste
consultar_usuarios()