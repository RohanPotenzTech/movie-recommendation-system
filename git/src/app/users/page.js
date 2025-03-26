'use client'

import React, { useEffect, useState } from 'react'

function Index() {
  const [input, setinput] = useState([])
  useEffect(() => {
    async function getData() {
      const data = await fetch('https://jsonplaceholder.typicode.com/users')
      const res = await data.json()
      setinput(res)
    }
    getData()
  }, [])
  
  return (
    <div>
      <h1>all files is below here</h1>
      {
        input.map((item) => (
          <div key={item.id}>
            <li>{item.name}</li>
            <br />
            <li>{item.username}</li>
            <br />
          </div>
        ))
      }
    </div>
  )
}

export default Index
