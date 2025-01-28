from discord import client

from summarizer import load_and_summarize_character_logs, load_and_summarize_character_logs_by_date


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('!resumo'):
        parts = message.content.split()
        character_name = parts[1] if len(parts) > 1 else None
        date = parts[2] if len(parts) > 2 else None

        if not character_name:
            await message.channel.send("Por favor, informe o nome do personagem.")
            return

        if date:
            summary = load_and_summarize_character_logs_by_date(character_name, date)
        else:
            summary = load_and_summarize_character_logs(character_name)

        if not summary:
            await message.channel.send(f"Nenhum log encontrado para o personagem {character_name}.")
            return

        await message.channel.send(summary)