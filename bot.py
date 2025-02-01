import asyncio
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from summarizer import load_and_summarize_character_logs

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

request_queue = asyncio.Queue()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


async def process_requests():
    while True:
        message, character_name = await request_queue.get()

        try:
            response = load_and_summarize_character_logs(character_name)
            await message.channel.send(f"Resumo de {character_name}:\n{response}")
        except Exception as e:
            await message.channel.send(f"Erro ao processar a requisição: {e}")
        finally:
            request_queue.task_done()


@bot.event
async def on_ready():
    print(f'Bot conectado como {bot.user}')
    asyncio.create_task(process_requests())


@bot.command(name="resumo")
async def resumo(ctx, character_name: str):
    await request_queue.put((ctx.message, character_name))
    await ctx.send(f"Requisição para resumo de {character_name} adicionada à fila. Aguarde...")


bot.run(DISCORD_TOKEN)
