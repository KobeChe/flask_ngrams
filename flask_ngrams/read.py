def FromMySql(conn,sql):
    try:
        cursor=conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        print(results)
        copy_rs = results
        return copy_rs
    except  Exception :print("查询失败")

def InsertMySql(conn,user_id):
	try:
		cursor=conn.cursor()
		sql1 = "insert into user_info1 (user_id) values (%s)"
		values = (user_id)
		cursor.execute(sql1,values)

	except   Exception :print("插入失败")


def CreateDict(user_id,lan,key,keyin,d1):
    #d={}
    
    if user_id in d1.keys():
        if lan in d1[user_id].keys():
            if key in d1[user_id][lan].keys():
                if keyin in d1[user_id][lan][key].keys():
                    d1[user_id][lan][key][keyin]=d1[user_id][lan][key][keyin]+1
                else:
                    d1[user_id][lan].setdefault(key,{})[keyin]=1
            else:
                d1[user_id][lan].setdefault(key,{})[keyin]=1


        else:
            d1[user_id].setdefault(lan,{})
            d1[user_id][lan].setdefault(key,{})[keyin]=1


    else:
        d1.setdefault(user_id,{})
        d1[user_id].setdefault(lan,{})
        d1[user_id][lan].setdefault(key,{})[keyin]=1
    return d1

        


