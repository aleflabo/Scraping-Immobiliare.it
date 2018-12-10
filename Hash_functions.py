def createNumber_permutation(listASCII):
    '''
    Convert the list of ASCII to a number. The goal is to have the same number no matter the permutations. 
    So if we have 2 password "ABC" and "ACB", the function generate the same number. So for that we are using
                for x in listASCII : ∏(99 + (x-32))
    We choose 99 because it's not too big soo the number is not too large and it's not too small to avoid colision.
    We remove 32 to our ASCII because the numbers from 0 to 32 of the ASCII table are not characters.
    
    Input: list of ASCII numberOuptut: our number (integer)'''
    # We remove 32 to our ASCII because the numbers from 0 to 32 of the ASCII table are not characters.
    # Then we 2 add it 99. Finally we multiply every result that we get together and divide it by 2.
    number = 1
    for i in listASCII:
    	number *= (99 + (i-32))
    return number
    
def myHashF_permutation(number):
    '''
    Hash our number. For this we are using a multiplication then a division. 
    			h(k) = (k*m)/a
	where m is the maximum value that the hash function can create and a is the maximum value that we can have.
    Input : our number (int)
    
    Output : our hashNumber (int)
    '''
    m = 3*109999990
    a = (10**40)
    hashNumber = int((number*m)/a)
    return hashNumber

def createNumber_withoutPermutation(listASCII):
    '''
    Convert the list of ASCII to a number. The goal is to have a number for each passwords. 
    So if we have 2 password "ABC" and "ACB", the function doesn't generate the same number.
    We remove 32 to our ASCII because the numbers from 0 to 32 of the ASCII table are not characters.
    
    Input: list of ASCII number
    
    Ouptut: integer
    '''
    number = int(''.join([str(n-32) for n in listASCII]))
    return number
    
def myHashF_withoutPermutation(number):
    '''
    Hash our number using the division method. h(k) = k mod m where k is our number and m is a constant
    which indicate the maximum value that the hash function can generate.
    
    Input : our number (int)
    
    Output : our hashNumber (int)
    '''
    m = (2**32)-1 #we want our hashNumber to be on 32bits
    hashNumber = number % m
    return hashNumber