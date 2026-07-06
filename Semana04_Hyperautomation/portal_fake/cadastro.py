# cadastro.py

usuarios = []

def cadastrar_usuario(nome, email):
    usuario = {
        "nome": nome,
        "email": email
    }

    usuarios.append(usuario)

    print("Usuário cadastrado com sucesso!")
    print(usuario)


# Teste
cadastrar_usuario("Maria Silva", "maria@email.com")

print("\nLista de usuários:")
print(usuarios)