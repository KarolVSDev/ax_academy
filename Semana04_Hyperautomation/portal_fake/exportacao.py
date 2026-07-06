# exportacao.py

usuarios = [
    {"nome": "Maria Silva", "email": "maria@email.com"},
    {"nome": "João Santos", "email": "joao@email.com"}
]

with open("usuarios.txt", "w", encoding="utf-8") as arquivo:
    for usuario in usuarios:
        arquivo.write(
            f"Nome: {usuario['nome']} | E-mail: {usuario['email']}\n"
        )

print("Dados exportados com sucesso!")