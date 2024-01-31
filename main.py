from app.gpt_chat import gpt_input,gpt_anwser
from app.front_end import front_end
import timeit

def main():
    
    start_time = timeit.default_timer()
    front_end()
    # Finaliza a contagem do tempo de resposta
    elapsed_time = timeit.default_timer() - start_time
    print(f"Executado em {elapsed_time} segundos")

if __name__ == "__main__":
    main()