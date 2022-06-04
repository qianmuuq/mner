import codecs
import numpy as np 

def get_chunks(seq, tags):
	"""
	tags:dic{'per':1,....}
	Args:
		seq: [4, 4, 0, 0, ...] sequence of labels
		tags: dict["O"] = 4
	Returns:
		list of (chunk_type, chunk_start, chunk_end)

	Example:
		seq = [4, 5, 0, 3]
		tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
		result = [("PER", 0, 2), ("LOC", 3, 4)]
	"""
	default = tags['O']
	idx_to_tag = {idx: tag for tag, idx in tags.items()}
	chunks = []
	# chunk_type用于判断是什么类型，LOC,PER
	chunk_type, chunk_start = None, None
	for i, tok in enumerate(seq):
		#End of a chunk 1 
		if tok == default and chunk_type is not None:
			# Add a chunk.
			chunk = (chunk_type, chunk_start, i)
			chunks.append(chunk)
			chunk_type, chunk_start = None, None

		# End of a chunk + start of a chunk!
		elif tok != default:
			# tok_chunk_class 判断是以B开头还是I开头
			# tok_chunk_type 判断是什么类型，PER,LOC
			tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
			if chunk_type is None:
				chunk_type, chunk_start = tok_chunk_type, i
			elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
				chunk = (chunk_type, chunk_start, i)
				chunks.append(chunk)
				chunk_type, chunk_start = tok_chunk_type, i
		else:
			pass
	# end condition
	if chunk_type is not None:
		chunk = (chunk_type, chunk_start, len(seq))
		chunks.append(chunk)
	return chunks

def get_chunk_type(tok, idx_to_tag):
	"""
	Args:
		tok: id of token, such as 4
		idx_to_tag: dictionary {4: "B-PER", ...}
	Returns:
		tuple: "B", "PER"
	"""
	tag_name = idx_to_tag[tok]
	tag_class = tag_name.split('-')[0]
	tag_type = tag_name.split('-')[-1]
	return tag_class, tag_type

# def run_evaluate(self, sess, test, tags):
def evaluate(labels_pred, labels,words,tags):

	"""
	words,pred, right: is a sequence, is label index or word index.
	Evaluates performance on test set
	Args:
		sess: tensorflow session
		test: dataset that yields tuple of sentences, tags
		tags: {tag: index} dictionary
	Returns:
		accuracy
		f1 score
		...
	"""

	#file_write = open('./test_results.txt','w')


	index = 0 
	sents_length = []

	accs = []
	correct_preds, total_correct, total_preds = 0., 0., 0.


	for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
		word_st = word_sent
		lab = lab
		lab_pred = lab_pred
		accs += [a==b for (a, b) in zip(lab, lab_pred)]
		lab_chunks = set(get_chunks(lab, tags))
		lab_pred_chunks = set(get_chunks(lab_pred, tags))
		correct_preds += len(lab_chunks & lab_pred_chunks)
		total_preds += len(lab_pred_chunks)
		total_correct += len(lab_chunks)

		#for i in range(len(word_st)):
				#file_write.write('%s\t%s\t%s\n'%(word_st[i],lab[i],lab_pred[i]))
		#file_write.write('\n')

	p = correct_preds / total_preds if correct_preds > 0 else 0
	r = correct_preds / total_correct if correct_preds > 0 else 0
	f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
	acc = np.mean(accs)

	#file_write.close()
	return acc, f1,p,r

def evaluate_each_class(labels_pred, labels,words,tags, class_type):
		#class_type:PER or LOC or ORG
		index = 0

		accs = []
		correct_preds, total_correct, total_preds = 0., 0., 0.
		correct_preds_cla_type, total_preds_cla_type, total_correct_cla_type = 0., 0., 0.

		for lab, lab_pred, word_sent in zip(labels, labels_pred, words):
				lab_pre_class_type = []
				lab_class_type=[]

				word_st = word_sent
				lab = lab
				lab_pred = lab_pred
				lab_chunks = get_chunks(lab, tags)
				lab_pred_chunks = get_chunks(lab_pred, tags)
				for i in range(len(lab_pred_chunks)):
						if lab_pred_chunks[i][0] ==class_type:
								lab_pre_class_type.append(lab_pred_chunks[i])
				lab_pre_class_type_c = set(lab_pre_class_type)

				for i in range(len(lab_chunks)):
						if lab_chunks[i][0] ==class_type:
								lab_class_type.append(lab_chunks[i])
				lab_class_type_c = set(lab_class_type)
				
				lab_chunksss = set(lab_chunks) 
				correct_preds_cla_type +=len(lab_pre_class_type_c & lab_chunksss)
				total_preds_cla_type +=len(lab_pre_class_type_c)
				total_correct_cla_type += len(lab_class_type_c)

		p = correct_preds_cla_type / total_preds_cla_type if correct_preds_cla_type > 0 else 0
		r = correct_preds_cla_type / total_correct_cla_type if correct_preds_cla_type > 0 else 0
		f1 = 2 * p * r / (p + r) if correct_preds_cla_type > 0 else 0

		return f1,p,r

def evaluate_bdui(labels_pred1,labels_pred2, labels,words,tags,id_tags,labels_pred3,labels_3,s_words):

	"""
	words,pred, right: is a sequence, is label index or word index.
	Evaluates performance on test set
	Args:
		sess: tensorflow session
		test: dataset that yields tuple of sentences, tags
		tags: {tag: index} dictionary
	Returns:
		accuracy
		f1 score
		...
	"""

	with open('./test_results.txt','w',encoding='utf-8') as f:


		index = 0
		sents_length = []

		accs = []
		correct_preds1,correct_preds2, total_correct, total_preds = 0., 0., 0.,0.


		for lab, lab_pred1,lab_pred2, word_sent,l_pred,l_3,s_word in zip(labels, labels_pred1,labels_pred2, words,labels_pred3,labels_3,s_words):
			word_st = word_sent
			lab = lab
			lab_pred1 = lab_pred1
			lab_pred2 = lab_pred2
			lab_chunks = set(get_chunks(lab, tags))
			lab_pred_chunks1 = set(get_chunks(lab_pred1, tags))
			correct_preds1 = len(lab_chunks & lab_pred_chunks1)

			lab_pred_chunks2 = set(get_chunks(lab_pred2, tags))
			correct_preds2 = len(lab_chunks & lab_pred_chunks2)

			if correct_preds1>correct_preds2:
				f.write('\n')
				for i,j in enumerate(word_st):
					f.write(word_st[i]+'\t'+id_tags[lab[i]]+'\t'+id_tags[lab_pred1[i]]+'\t'+id_tags[lab_pred2[i]]+'\n')
				f.write('\n')
				for i,j in enumerate(s_word):
					f.write(s_word[i]+'\t'+id_tags[l_3[i]]+'\t'+id_tags[l_pred[i]]+'\n')
				f.write('\n')

	return 0

def evaluate_bdui2(labels_pred1,labels_pred2, labels,words,tags,id_tags):

	"""
	words,pred, right: is a sequence, is label index or word index.
	Evaluates performance on test set
	Args:
		sess: tensorflow session
		test: dataset that yields tuple of sentences, tags
		tags: {tag: index} dictionary
	Returns:
		accuracy
		f1 score
		...
	"""

	with open('./test_results.txt','w',encoding='utf-8') as f:


		index = 0
		sents_length = []

		accs = []
		correct_preds1,correct_preds2, total_correct, total_preds = 0., 0., 0.,0.


		for lab, lab_pred1,lab_pred2, word_sent in zip(labels, labels_pred1,labels_pred2, words):
			word_st = word_sent
			lab = lab
			lab_pred1 = lab_pred1
			lab_pred2 = lab_pred2
			lab_chunks = set(get_chunks(lab, tags))
			lab_pred_chunks1 = set(get_chunks(lab_pred1, tags))
			correct_preds1 = len(lab_chunks & lab_pred_chunks1)

			lab_pred_chunks2 = set(get_chunks(lab_pred2, tags))
			correct_preds2 = len(lab_chunks & lab_pred_chunks2)

			if correct_preds1<correct_preds2:
				f.write('\n')
				for i,j in enumerate(word_st):
					f.write(word_st[i]+'\t'+id_tags[lab[i]]+'\t'+id_tags[lab_pred1[i]]+'\t'+id_tags[lab_pred2[i]]+'\n')
				f.write('\n')

	return 0


if __name__ == '__main__':
		max_sent=10
		tags = {'0':0,
		'B-PER':1, 'I-PER':2,
		'B-LOC':3, 'I-LOC':4,
		'B-ORG':5, 'I-ORG':6,
		'B-OTHER':7, 'I-OTHER':8,
		'O':9}
		labels_pred=[
								[9,9,9,1,3,1,2,2,0,0],
								[9,9,9,1,3,1,2,0,0,0]
		]
		labels = [
						[9,9,9,9,3,1,2,2,0,0],
						[9,9,9,9,3,1,2,2,0,0]
						]
		words = [
						[0,0,0,0,0,3,6,8,5,7],
						[0,0,0,4,5,6,7,9,1,7]
						]
		id_to_vocb = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j'}
		# new_words = []
		# for i in range(len(words)):
		# 	sent = []
		# 	for j in range(len(words[i])):
		# 		sent.append(id_to_vocb[words[i][j]])
		# 	new_words.append(sent)
		# class_type = 'PER'
		# acc, f1,p,r = evaluate(labels_pred, labels,new_words,tags)
		# print(p,r,f1)
		# f1,p,r = evaluate_each_class(labels_pred, labels,new_words,tags, class_type)
		# print(p,r,f1)

		acc, f1, p, r = evaluate(labels_pred, labels, words, tags)
		print(acc)