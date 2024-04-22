import pandas as pd
test_texts=pd.read_pickle('test_texts.pkl')
test_summaries=pd.read_pickle('test_summaries.pkl')
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
def load_model(model_path):
    device='cuda'
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    return model

def load_tokenizer(tokenizer_path):
    device='cuda'
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path,device=device)

    return tokenizer

def generate_text(model_path, max_length,num_samples,test_texts):

    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    generated_summaries=[]
    for i in range(num_samples):
        gen=set()
        iters=0
        while (len(gen)<3 and iters<10):
            sequence =f'Query: {test_texts.iloc[i]} TL;DR Summary:'
            # try:
            ids = tokenizer.encode(f'{sequence}', return_tensors='pt').to('cuda')

            final_outputs = model.generate(
                ids,
                do_sample=True,
                max_new_tokens=max_length,
                pad_token_id=model.config.eos_token_id,
                top_k=50,
                top_p=0.95,
            )
            bruh=tokenizer.decode(final_outputs[0], skip_special_tokens=True)
            try:   
                bruh=bruh.split('Summary: ')[1]
                bruh=bruh.split('Query')[0]
            except:
                gen.add("Error")
                break
            gen.add(bruh)
            # except:
            #     gen.add("Error")
            #     break
            iters+=1
        generated_summaries.append(gen)


    return generated_summaries


generated_summaries=[]
num_samples=20
model_path = 'gpt2-tldr'

max_length = 15
generated_summaries=generate_text(model_path,max_length,num_samples,test_texts)
#calculate rouge score
from rouge import Rouge
rouge = Rouge()
rogue_scores=[] 


for i in range(num_samples):
    gand=list(generated_summaries[i])
    max=rouge.get_scores(gand[0], test_summaries.iloc[i])
    max_val=max[0]['rouge-1']['p']
    for j in range(1,len(gand)):
        scores = rouge.get_scores(gand[j], test_summaries.iloc[i])
        new_val=scores[0]['rouge-1']['p']
        if(new_val>max_val):
            max=scores
    rogue_scores.append(max)

#give the average rouge score
rouge_1_prec=0
rouge_1_recall=0
rouge_1_f=0
rouge_2_prec=0
rouge_2_recall=0
rouge_2_f=0
rouge_l_prec=0
rouge_l_recall=0
rouge_l_f=0
for i in range(num_samples):
    rouge_1_prec+=rogue_scores[i][0]['rouge-1']['p']
    rouge_1_recall+=rogue_scores[i][0]['rouge-1']['r']
    rouge_1_f+=rogue_scores[i][0]['rouge-1']['f']
    rouge_2_prec+=rogue_scores[i][0]['rouge-2']['p']
    rouge_2_recall+=rogue_scores[i][0]['rouge-2']['r']
    rouge_2_f+=rogue_scores[i][0]['rouge-2']['f']
    rouge_l_prec+=rogue_scores[i][0]['rouge-l']['p']
    rouge_l_recall+=rogue_scores[i][0]['rouge-l']['r']
    rouge_l_f+=rogue_scores[i][0]['rouge-l']['f']
rouge_1_prec=rouge_1_prec/num_samples
rouge_1_recall=rouge_1_recall/num_samples
rouge_1_f=rouge_1_f/num_samples
rouge_2_prec=rouge_2_prec/num_samples
rouge_2_recall=rouge_2_recall/num_samples
rouge_2_f=rouge_2_f/num_samples
rouge_l_prec=rouge_l_prec/num_samples
rouge_l_recall=rouge_l_recall/num_samples
rouge_l_f=rouge_l_f/num_samples
print('rouge-1 precision:',rouge_1_prec)
print('rouge-1 recall:',rouge_1_recall)
print('rouge-1 f:',rouge_1_f)
print('rouge-2 precision:',rouge_2_prec)
print('rouge-2 recall:',rouge_2_recall)
print('rouge-2 f:',rouge_2_f)
print('rouge-l precision:',rouge_l_prec)
print('rouge-l recall:',rouge_l_recall)
print('rouge-l f:',rouge_l_f)
