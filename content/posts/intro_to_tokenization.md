---
title: "Intro to Tokenization using BPE - BytePair Encoding"
date: 2026-05-31
draft: false
---

in the previous blog about running evals on language models, i left a few tasks for future work. but as i start reading, it felt like it's important to learn tokenization before trying to figure out the answers to all those unanswered questions in my previous blog. so, here's me, trying to understand tokenization. my plan is to understand everything about LLMs - to be able to create my own LLM and run inference in case i have the resources.

end goal: **implement BytePair Encoding at a small scale**

so, tokenization? what exactly are tokens? you can vaguely consider tokens as words, but not exactly. for example, "ing" might be a token but it isn't really a meaningful word. a token is the smallest unit in LLMs. this is why you have token limits and not word-limits/sentence-limits/compute-limits or whatever, in LLM subscriptions. also, here's [karpathy's tokenization lecture](https://www.youtube.com/watch?v=zduSFxRajkE) that i watched before this blog. that was my starting point to learn tokenization. 

basically, you take a large amount of text. you find the most repeating subpart of the text and then list down those parts. you give a *token number* to these most repeating things. it's only after this, you get a list of numbers (token numbers), and you can run some math operations on these numbers i.e., train the model to finally get a language model which will respond to you in terms of "tokens" that we created during tokenization. 

and so the first question is - "where do these numbers come from". in practice, people use [unicodes](https://home.unicode.org/) at first. these are the publicly accepted *codes* that every device uses so that when your computer is communicating with another computer, nothing breaks. it's just a set of standards created for optimal public use. it's worth spending some time on unicodes to understand how different languages, including emojis, can be displayed the exact same way in every single device. here are two links to understand unicodes better - [unicode intro by Nathan Reed](https://www.reedbeta.com/blog/programmers-intro-to-unicode/) & [utf8 everywhere manifesto](https://utf8everywhere.org/) 

a very short intro here regarding unicodes. unicode guys decided that they're gonna create a standard set to include every language, every ideograph, every symbol that existed and also something that might exist in future. i guess the first major standardization was ASCII and anyone who wants to create any standards would choose to support ASCII characters as backward compatibility. eventually, they came up with 17 planes. the first plane is plane-0 i.e., BMP - Basic Multilingual PLane.

![reedbeta screenshot about plane-0,1 and others](/2/reedbeta_ss_1.png)

smallest unit of unicodes is "codepoint", not exactly the same as "character". in the entire codespace, each codepoint is represented by a hexadecimal with the prefix "U+" like  U+0041, “A”, latin capital letter A. and when we want to train, we prefer a decimal number instead of a hexadecimal one, and this decimal number is used as a token for a *character*. google online and you'll see that "61" value will represent "=". and then when you have a large text, the text can be converted into a numerical form based on UTF8 encoding. UTF8 is the preferred standard for optimal usage. now what we've converted our text into a list of numbers, that a computer can work on, we are ready to implement BytePair Encoding. 

**Understanding Byte Pair Encoding**

we are here, assuming that, if text can be represented by numbers, we can perform some statistical manipulation on these numbers to create a machine learning model that can generate a response to our textual query based on the entire text that it's been trained on. and so we think about all the ways we can represent text in a numerical form. 

![https://platform.openai.com/tokenizer screenshot](/2/tokenizer_gpt_ss.png)

you can see how the current version of tokenization works in openai models. so, why don't we just use a regular dictionary and assign *token numbers* to each word and call it a day? well, there are many words that won't exist in a dictionary because it's so *standardized*. terms like "imo", "tbh" won't find a place in a regular dictionary. so do we create and maintain an informal dictionary that has the meanings of all existing words? nope, that's kinda impossible or at least a bad design imo, don't you think? well, why don't we simply use a token per unicode? that'd mean that we won't ever have to worry about new unseen tokens ever again. but, it implies that we must be ready for huge increment in the token length. when a token is represented as a number, we perform some calculations and generate more tokens. if every token now becomes 4-5 new tokens on average, because we chose character level tokenization instead of word-styled tokenization, it'll have significant strain on the computations that it'll be next to impossible to work with, using even our current advanced GPUs. in fact, the reason why LLMs managed to work in this particular age is because all the GPU improvements that was done for gamers gave people a breakthrough in real life computations that are possible. so, yes, it seems like choosing a word-styled tokenization is a better choice. 

at this point, we still haven't figured out what kinda tokenization we're going to choose in order to balance both the computations and simplicity. here comes BPE - *Byte Pair Encoding*, where we just iterate through the entire text we have and then notice which pair is repeating the maximum number of times. now, when we realize that a certain pair is repeating too many times, it makes sense that we assign a *token number* to this pair. for instance, let us consider the term "banana". the pairs we get here are - { (b,a), (a,n), (n,a), (a,n), (n,a) }. if we have to represent the word "banana" using numbers, we can first assign a number to each character. and then when we notice that (a,n) is being repeated twice, we find it better to represent the combination (a,n) with a single token number. sure, you can do the same with (n,a) pair too, as it has 2 repetitions. so the word "banana" can now become something like "bxxa" which is shorter than the original word and all we have to do is remember what pairing results in "x". in fact, we can even go a step further and call it "bya" by replacing "xx" with "y". 

all we're doing here is :
- iterate through text to find most common pair
- merge the most common pair into a new *token* which decreases the token count. 
- repeat this until we are satisfied with the number of tokens.

and so here's my draft version of this process in python. 

```python

# draft 0: 
# -> iterate over text once and extract highest repeating byte-pair

# dictionary to store byte-pair counts
counts = {}
t = """given a text, iterate over it while noting down the "byte-pair" count and then merge the pair with highest count"""
for i in range(len(t)-1):
    pair = (t[i], t[i+1])
    if t[i+1]!=" ":
        counts[pair] = counts.get(pair, 0) + 1

print(len(counts))
best_pair = (max(counts, key=counts.get))
print(counts[best_pair])
print(best_pair)

# OR best_pair, best_count = max(counts.items(), key=lambda x: x[1])

# draft 0 issues: 
# it includes a space after characters. but let us only include space before the characters to keep tokens consistent;
```
as you can see, i noted down an issue in the end. the problem is that the most repeating pair happened to be "e ", i.e., "e" followed by a space " ". this is a significant issue. spaces exist before and after the words. now if you consider both these spaces to be part of tokens, then when two tokens are generated next to each other, you might end up double spaces between two words. this isn't desirable. also, if you look at the gpt tokenization, you can find that a space is only included for a word in the beginning and not at the end. later, when you are displaying the text to user, you can manually avoid space whenever needed, like at the starting of a paragraph. in this code block, i just used a small text. 

here, we just took care of only one most repeating pair. let us move forward to finding more pairs. as we find a pair, we can store that info in a dictionary to represent all the merges that we did. but as we have multiple merges, we obviously need to represent these merges with a new token number everytime. (0->255) is already blocked by the *standards* that we already decided to use. we already decided to represent "A" with 65, "B" with 66, etc. so, we'll assign new token numbers starting from 256 and incrementing one for every pair merged. in this below code in particular, we only try merging once before we go for multiple merges. 

```python
# draft 1:
# -> find the max repeating bytepair and add it to "merges" dictionary
# -> create a new list of tokens with this updated bytepair reflecting a new token in the pair's place

new_token = 256
merges = {}
val1 = (len(current_text_tokens))
print(val1)

def merge_one(tokens_list, merges_dict, new_token):
    counts = {}
    for i in range(len(tokens_list)-1):
        if tokens_list[i] == 61 or tokens_list[i]==32: # 61 is "=" and 32 is " "; too many 61s in that gencode; it was messing things, so i removed it;
            continue
        pair = (tokens_list[i], tokens_list[i+1])
        counts[pair] = counts.get(pair, 0) + 1
    # best_pair = max(counts, key = counts.get)
    best_pair, best_count = max(counts.items(), key=lambda x: x[1])
    print(best_count)
    print(best_pair)
    merges_dict[best_pair] = new_token

    tokens_list_new = []
    i = 0
    while i<len(tokens_list):
        if((i+1<len(tokens_list)) and (tokens_list[i], tokens_list[i+1]) == best_pair):
            tokens_list_new.append(new_token)
            i +=2
        else:
            tokens_list_new.append(tokens_list[i])
            i +=1
    return tokens_list_new

val2 = (len(merge_one(current_text_tokens, merges, new_token)))
print(val2)
print(val1-val2)
```

as you can see, i avoided a couple of tokens - 61 and 32. i took all the text from my previous blog and it had too many "=" as part of comments in the python code. i did this just to see if everything's working and if i could find other repeating pairs. i did. also, i printed the before and after counts of token list (text) to see if my merging is working as expected or not. 

and then here's my code for merging multiple times by iterating through the text over and over again while updating the text_tokens as per the previously created merging. i'm choosing to go for just 10 merges. also, forgot to mention that all of this can be done locally on your device as this isn't some high compute stuff. only when we really get into LLMs training do we need to use GPUs. 

```python

with open("test_text3.txt", "r", encoding="utf-8") as file:
    text = file.read()

current_text_tokens = list(text.encode("utf-8")) 

print(len(current_text_tokens))

# draft 2: 
# -> iterate through the text and find max repeating bytepair tokens
# -> create a new list of tokens with this updated bytepair reflecting a new token in the pair's place
# -> decide the number of merges you wanna have and then run a for loop on that merge function
# -> update the new token everytime a merge happens

def merge_one(tokens_list, merges_dict, new_token):
    counts = {}
    for i in range(len(tokens_list)-1):
        pair = (tokens_list[i], tokens_list[i+1])
        counts[pair] = counts.get(pair, 0) + 1
    # best_pair = max(counts, key = counts.get)
    best_pair, best_count = max(counts.items(), key=lambda x: x[1])
    print(best_count)
    # print(best_pair)
    merges_dict[best_pair] = new_token

    tokens_list_new = []
    i = 0
    while i<len(tokens_list):
        if((i+1<len(tokens_list)) and (tokens_list[i], tokens_list[i+1]) == best_pair):
            tokens_list_new.append(new_token)
            i +=2
        else:
            tokens_list_new.append(tokens_list[i])
            i +=1
    return tokens_list_new


def merge_many(tokens_list, merges_dict, new_token, num_merges):
    updated_token_list = tokens_list

    for i in range(num_merges):
        updated_token_list = merge_one(updated_token_list, merges_dict, new_token)
        new_token +=1
    return updated_token_list

num_merges = 10
new_token = 256
merges_dict = {}
final_token_list = merge_many(current_text_tokens, merges_dict, new_token, num_merges)

print(len(final_token_list))
```

in this case, i wanted to avoid those extra specifications like removing certain unicodes. so, i chose Mary Shelly's "Frankenstein" and it turned out to be perfect text for this because when i looked at the difference between final token list and initial token list, the number of merges were perfect. here below is the output for this code :
```
440623
11867
9113
8978
6007
5996
5796
5614
5537
4727
4647
372341
```

i guess this ends our BPE implementation. sure, you can include the encoding and decoding functionalities on top of this. but this isn't perfect. there are still so many regex functions we need to tokenize at the production level. for instance, how would you tokenize numbers? in karpathy's video, he mentions how sometimes, the tokenizer divides a 4 digit number as (2,2) or as (1,3) or maybe (3,1) and all these differences will definitely impact. remember the last [evals blog](https://vkaschism.github.io/posts/arithmetic_eval_slm_1/) where the model couldn't answer perfectly when the digits became too many, yep, all those things are to be considered in the next phase. 

for the next blog, i'll be working on a near-perfect tokenizer and compare it with gpt tokenizer. karpathy somehow showed that his minbpe tokenizer gives similar token values as that of openai's tokenizer. i didn't spend any time on that but will, soon. also, i'll look at telugu language tokenization. while it does seem like the inherent complex grammar structure of telugu might be of some use, the truth is that we don't really care about any such thing in BPE. whatever grammar structure exists in a language will be reflected in the tokenizer based on *training* text we give it due to **statistics & probability**. and yes, soon, i need to learn the math behind it. 

one more thing, i kinda find it odd that we call this thing "training" because whatever the word "training" meant to me when i looked at all those machine learning things, this doesn't even come close to that in any way. yet, i guess, some people do call this thing **training**. anyways, it doesn't really matter what you call it. 
